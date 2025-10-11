from VQGAN.taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from VQGAN.taming.models.vqgan import VQModel

def creat_model(opt):
    return VQModel(opt)