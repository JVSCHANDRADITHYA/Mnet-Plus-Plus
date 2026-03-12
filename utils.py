import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(1 / mse)


def save_samples(epoch, inp, out, gt):

    grid = torch.cat([inp, out, gt], dim=0)

    vutils.save_image(
        grid,
        f"samples/epoch_{epoch}.png",
        nrow=inp.shape[0],
        normalize=True
    )