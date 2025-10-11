import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import scipy
import numpy as np
import os
from diffusion import create_diffusion
from taming.modules.diffusionmodules.model import Encoder
from modules.distributions.distributions import DiagonalGaussianDistribution
from DiT_model import SwinIR
from DiT_model import DiT
import data as Data
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel as EMA

def set_device(device, x):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None:
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x

def load_vae(root, ckp, device):
    ddconfig = {
        'double_z': True,
        'z_channels': 4,
        'resolution': 192,
        'in_channels': 2,
        'c_channels': 2,
        'out_ch': 1,
        'ch': 32,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 1,
        'attn_resolutions': [16],
        'dropout': 0.2,
    }

    cdconfig = {
        'double_z': True,
        'z_channels': 4,
        'resolution': 192,
        'in_channels': 2,
        'c_channels': 2,
        'out_ch': 1,
        'ch': 32,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 1,
        'attn_resolutions': [16],
        'dropout': 0.2,
    }

    encoder = set_device(device, Encoder(**ddconfig))
    cond_encoder = set_device(device, Encoder(**cdconfig))
    decoder = SwinIR(img_size=192, patch_size=1, in_chans=2, out_chans=1, embed_dim=32,
                     depths=(3, 3, 3, 3), num_heads=(4, 4, 4, 4), z_chans=4, window_size=8).to(device)

    checkpoint = torch.load(os.path.join(root, 'checkpoints', f'e_{ckp}.pth'), map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    # cond_encoder.load_state_dict(checkpoint['cond_enc'])

    print(f'Loaded {ckp} VAE checkpoint!')

    encoder.eval()
    decoder.eval()
    return encoder, decoder, cond_encoder

class MultiChannelSobelLoss(torch.nn.Module):
    def __init__(self):
        super(MultiChannelSobelLoss, self).__init__()
        # Sobel kernel for X and Y directions
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32) /8
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32) /8
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, output, target):
        B, C, H, W = output.shape
        kx = self.sobel_x.to(output.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        ky = self.sobel_y.to(output.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        grad_px = F.conv2d(output, kx, padding=1, groups=C)
        grad_py = F.conv2d(output, ky, padding=1, groups=C)
        grad_tx = F.conv2d(target, kx, padding=1, groups=C)
        grad_ty = F.conv2d(target, ky, padding=1, groups=C)

        loss = F.l1_loss(grad_px, grad_tx) + F.l1_loss(grad_py, grad_ty)
        return loss * 0.5

class CharbonnierLoss(torch.nn.Module):
    """CharbonnierLoss."""
    def __init__(self, eps):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        error = torch.sqrt((pred - target)**2 + self.eps)
        loss = torch.mean(error)
        return loss

if __name__ == "__main__":

    Epoch = 100
    start_epoch = 0
    global_step = 0
    device = ''
    batch_size = 12
    ckp_root = ""
    res_root = ""
    vae_root = ""
    vae_ckp = 100
    checkpoint = 0
    warmup_epoch = 30

    gpu_list = [int(id) for id in device.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    print('export CUDA_VISIBLE_DEVICES=' + device)
    device = torch.device('cuda')

    encoder, decoder, cond_encoder = load_vae(vae_root, vae_ckp, device)
    c_loss = CharbonnierLoss(1e-4).to(device)
    sobel_loss = MultiChannelSobelLoss().to(device)

    model =DiT(img_size=24, in_chans=8, out_chans=8, embed_dim=32,
                     depths=(3, 3, 3, 3), num_heads=(4, 4, 4, 4), z_chans=8, window_size=24).to(device)

    model = set_device(device, model)

    ema_modules = {
        "model": EMA(model, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
    0.999 * averaged_model_parameter + (1 - 0.999) * model_parameter),
        "cond_encoder": EMA(cond_encoder, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
    0.999 * averaged_model_parameter + (1 - 0.999) * model_parameter),
        "decoder": EMA(decoder, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
    0.999 * averaged_model_parameter + (1 - 0.999) * model_parameter)
    }

    if checkpoint != 0:
        # load checkpoints
        checkpoint = torch.load(os.path.join(ckp_root, 'checkpoints', f'e_{checkpoint}.pth'), map_location=device)
        model.load_state_dict(checkpoint['model'])
        cond_encoder.load_state_dict(checkpoint['cond_encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        ema_modules['model'].module.load_state_dict(checkpoint['ema_model'])
        ema_modules['cond_encoder'].module.load_state_dict(checkpoint['ema_cond_encoder'])
        ema_modules['decoder'].module.load_state_dict(checkpoint['ema_decoder'])
        start_epoch = checkpoint['epoch']+1
        global_step = checkpoint['global_step']
        print(f'Loaded {start_epoch} DiT checkpoint!')

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=25, scale=500, beta_scale=0.8)

    train_loader = Data.creat_trainloader(batch_size)

    base_lr = 1e-4
    opt = torch.optim.Adam((list(model.parameters()) + list(cond_encoder.parameters()) + list(decoder.parameters())), lr=base_lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(opt, T_max=(Epoch-max(start_epoch, warmup_epoch)) * len(train_loader), eta_min=1e-5)

    score_opt = torch.optim.Adam((list(model.parameters()) + list(cond_encoder.parameters())),
                           lr=base_lr, betas=(0.5, 0.9))

    for current_epoch in range(start_epoch, Epoch):

        encoder.eval()
        model.train()
        cond_encoder.train()
        decoder.train()

        train_loop = tqdm(train_loader, desc='Training epoch#{}'.format(current_epoch))
        avg_loss = 0
        for step, (train_data, name) in enumerate(train_loop):
            global_step += 1
            clean = set_device(device, train_data['clean'])
            noisy = set_device(device, train_data['noisy'])
            mr = set_device(device, train_data['mr'])
            input = cond_encoder(torch.cat((noisy, mr), dim=1))
            with torch.no_grad():
                z_gt = encoder(torch.cat((clean, mr), dim=1))

            if current_epoch <= warmup_epoch:
                t = torch.randint(0, diffusion.num_timesteps, (clean.shape[0],), device=device)
                model_kwargs = dict(z=input)

                loss_dict = diffusion.training_losses(model, z_gt, t, model_kwargs)
                loss = loss_dict['loss'].mean()

                score_opt.zero_grad()
                loss.backward()
                score_opt.step()
                current_loss = np.squeeze(np.squeeze(loss.clone().data.cpu())).numpy()

                avg_loss += (current_loss - avg_loss) / (step + 1)
                train_loop.set_postfix(score_Loss=avg_loss, step=global_step)

            else:
                t = (diffusion.num_timesteps-1)*torch.ones((clean.shape[0],), device=device)
                z = diffusion.q_sample(z_gt, t.int())
                model_kwargs = dict(z=input)
                z_pred = diffusion.p_sample_loop_training(model, z.shape, z, clip_denoised=False,
                                                          model_kwargs=model_kwargs)

                latent_loss = c_loss(z_pred, z_gt) + 0.05 * sobel_loss(z_pred, z_gt)

                posterior_pred = DiagonalGaussianDistribution(z_pred)
                z_pred = posterior_pred.sample()

                posterior = DiagonalGaussianDistribution(z_gt)
                z_gt = posterior.sample()

                sample = decoder(torch.cat((noisy, mr), dim=1), z_pred)

                img_loss = c_loss(sample, clean) + 0.05 * sobel_loss(sample, clean)

                loss = img_loss + 0.05 * latent_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
                current_loss = np.squeeze(np.squeeze(loss.clone().data.cpu())).numpy()

                avg_loss += (current_loss - avg_loss) / (step + 1)
                train_loop.set_postfix(Loss=avg_loss, step=global_step)
                scheduler.step()

                ema_modules['model'].update_parameters(model)
                ema_modules['cond_encoder'].update_parameters(cond_encoder)
                ema_modules['decoder'].update_parameters(decoder)

        weights_file = os.path.join(res_root, 'checkpoints', f'e_{current_epoch}.pth')
        torch.save({
            'epoch': current_epoch,
            'global_step': global_step,
            'cond_encoder': cond_encoder.state_dict(),
            'model': model.state_dict(),
            'decoder': decoder.state_dict(),
            'ema_model': ema_modules['model'].module.state_dict(),
            'ema_cond_encoder': ema_modules['cond_encoder'].module.state_dict(),
            'ema_decoder': ema_modules['decoder'].module.state_dict()
        }, weights_file)