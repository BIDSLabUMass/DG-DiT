import torch
import torch.nn as nn

from autoencoder.taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


def kl(mean, logvar, deterministic=False, other=None):
    var = torch.exp(logvar)
    if deterministic:
        return torch.Tensor([0.])
    else:
        if other is None:
            return 0.5 * torch.sum(torch.pow(mean, 2)
                                   + var - 1.0 - logvar,
                                   dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                torch.pow(mean - other.mean, 2) / other.var
                + var / other.var - 1.0 - logvar + other.logvar,
                dim=[1, 2, 3])

class myLPIPSWithDiscriminator(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors_mean, posteriors_logvar, weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # kl_loss = posteriors.kl()
        kl_loss = kl(posteriors_mean, posteriors_logvar)
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        return loss


