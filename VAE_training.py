import torch, os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.optim as optim
from tqdm import tqdm
import data as Data
from taming.modules.diffusionmodules.model import Encoder
from DiT_model import SwinIR
from modules.distributions.distributions import DiagonalGaussianDistribution
from modules.losses.mycontperceptual import myLPIPSWithDiscriminator
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    device = ''
    batch_size = 16
    checkpoint = 0
    start_epoch = 0
    ckp_root = ""
    res_root = ""

    train_loader = Data.creat_trainloader(batch_size)

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
        'dropout': 0,
    }

    encoder = Encoder(**ddconfig).to(device)
    decoder = SwinIR(img_size=192, patch_size=1, in_chans=2, out_chans=1, embed_dim=32,
                     depths=(3, 3, 3, 3), num_heads=(4, 4, 4, 4), z_chans=4, window_size=8).to(device)
    VAEloss = myLPIPSWithDiscriminator(kl_weight=0.000001).to(device)

    if checkpoint != 0:
        # save checkpoints
        ckp = torch.load(os.path.join(ckp_root, 'checkpoints', f'e_{checkpoint}.pth'), map_location=device)
        encoder.load_state_dict(ckp['encoder'])
        decoder.load_state_dict(ckp['decoder'])
        start_epoch = checkpoint + 1
        print(f'Loaded {checkpoint} checkpoint!')

    LR = 1e-4
    ae_params = (list(encoder.parameters()) + list(decoder.parameters()))
    optimizer = optim.Adam(ae_params, lr=LR, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=(Epoch - start_epoch) * len(train_loader),
                                  eta_min=1e-5)

    c_loss = CharbonnierLoss(1e-4).eval().to(device)

    for epoch in range(start_epoch, Epoch):

        avg_loss = 0
        train_loop = tqdm(train_loader)
        encoder.train()
        decoder.train()
        for step, (train_data, _) in enumerate(train_loop):
            noisy = train_data['noisy'].to(device)
            clean = train_data['clean'].to(device)
            mr = train_data['mr'].to(device)
            # noisy, clean, mr = random_crop_triplet(noisy, clean, mr, patch)

            clean_input = torch.cat((clean, mr), dim=1)
            noisy_input = torch.cat((noisy, mr), dim=1)

            posterior = DiagonalGaussianDistribution(encoder(clean_input))
            z = posterior.sample()
            denoised = decoder(noisy_input, z)

            loss = VAEloss(clean, denoised, posterior.mean, posterior.logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            avg_loss += (current_loss - avg_loss) / (step + 1)
            train_loop.set_postfix(Loss=avg_loss)
            scheduler.step()

        weights_file = f'{res_root}/checkpoints/e_{epoch}.pth'
        torch.save({'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict()}, weights_file)