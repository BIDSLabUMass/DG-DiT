import torch
import torch.nn.functional as F
import torch.nn as nn

from VQGAN.main import instantiate_from_config

from VQGAN.taming.modules.diffusionmodules.model import Encoder, Decoder
from VQGAN.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def dit_encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def dit_decode(self, h):
        quant, _, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def dit_forward(self, input):
        h = self.dit_encode(input)
        dec = self.dit_decode(h)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, target):
        pet = batch[target]
        # x = torch.cat([pet, mr], dim=1)
        x = pet.to(memory_format=torch.contiguous_format)
        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight