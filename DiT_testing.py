import torch
import scipy
import numpy as np

import os
from diffusion import create_diffusion
from taming.modules.diffusionmodules.model import Encoder
from modules.distributions.distributions import DiagonalGaussianDistribution
from DiT_model import SwinIR
from DiT_model import DiT
import data as Data

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

    print(f'Loaded {ckp} VAE checkpoint!')

    encoder.eval()
    decoder.eval()
    return encoder, decoder, cond_encoder


if __name__ == "__main__":

    device = ''
    gpu_list = [int(id) for id in device.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    print('export CUDA_VISIBLE_DEVICES=' + device)
    device = torch.device('cuda')

    vae_root = ""
    vae_ckp = 100

    res_root = ""
    checkpoint = 100
    model = DiT(img_size=24, in_chans=8, out_chans=8, embed_dim=32,
                       depths=(3, 3, 3, 3), num_heads=(4, 4, 4, 4), z_chans=8, window_size=24).to(device)

    model = set_device(device, model)

    encoder, decoder, cond_encoder = load_vae(vae_root, vae_ckp, device)

    if checkpoint != 0:
        # load checkpoints
        checkpoint = torch.load(os.path.join(res_root, 'checkpoints', f'e_{checkpoint}.pth'), map_location=device)
        model.load_state_dict(checkpoint['ema_model'])
        cond_encoder.load_state_dict(checkpoint['ema_cond_encoder'])
        decoder.load_state_dict(checkpoint['ema_decoder'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f'Loaded {start_epoch} DiT checkpoint!')

    model.eval()
    encoder.eval()
    decoder.eval()
    cond_encoder.eval()

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=25, scale=500, beta_scale=0.2)
    scanners = ['HRRT', 'Biograph']
    tracers = ['FDG', 'AV1451', 'AV45']

    for scanner in scanners:

        for tracer in tracers:
            test_loader = Data.creat_testloader('test', scanner, tracer)

            for step, (val_data, name) in enumerate(test_loader):
                if step % 100 == 0:
                    print(f"{step} image in {scanner} scanner, {tracer} tracer")

                with torch.no_grad():
                    name = name[0]
                    clean = set_device(device, val_data['clean'])
                    noisy = set_device(device, val_data['noisy'])
                    mr = set_device(device, val_data['mr'])
                    img_input = torch.cat((noisy, mr), dim=1)

                    input = cond_encoder(img_input)

                    z = torch.randn(input.shape[0], 8, 24, 24, device=device)
                    model_kwargs = dict(z=input)
                    z_pred = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=False,
                                                     model_kwargs=model_kwargs)
                    posterior_pred = DiagonalGaussianDistribution(z_pred)
                    z_pred = posterior_pred.sample()
                    sample = decoder(img_input, z_pred)

                    posterior = DiagonalGaussianDistribution(encoder(torch.cat((clean, mr), dim=1)))
                    z_gt = posterior.sample()

                    clean = np.squeeze(clean.data.cpu())
                    noisy = np.squeeze(noisy.data.cpu())
                    mr = np.squeeze(mr.data.cpu())
                    denoised = np.squeeze(sample.data.cpu())
                    z_gt = np.squeeze(z_gt.data.cpu())
                    z_pred = np.squeeze(z_pred.data.cpu())

                    val_file = os.path.join(res_root, 'test', scanner, tracer, f'{name}')
                    scipy.io.savemat(val_file, {'clean': clean.numpy(), 'noisy': noisy.numpy(),
                                                'mr': mr.numpy(), 'denoised': denoised.numpy(),
                                                'z_gt': z_gt.numpy(), 'z_pred':z_pred.numpy()})