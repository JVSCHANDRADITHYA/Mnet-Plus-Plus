import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1,0,-1],
                                [2,0,-2],
                                [1,0,-1]], dtype=torch.float32)

        sobel_y = torch.tensor([[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]], dtype=torch.float32)

        self.weight_x = sobel_x.view(1,1,3,3)
        self.weight_y = sobel_y.view(1,1,3,3)

    def forward(self, pred, target):

        weight_x = self.weight_x.to(pred.device)
        weight_y = self.weight_y.to(pred.device)

        pred_gray = pred.mean(1, keepdim=True)
        tgt_gray = target.mean(1, keepdim=True)

        pred_x = F.conv2d(pred_gray, weight_x, padding=1)
        pred_y = F.conv2d(pred_gray, weight_y, padding=1)

        tgt_x = F.conv2d(tgt_gray, weight_x, padding=1)
        tgt_y = F.conv2d(tgt_gray, weight_y, padding=1)

        return F.l1_loss(pred_x, tgt_x) + F.l1_loss(pred_y, tgt_y)
    
    
class FFTLoss(nn.Module):

    def forward(self, pred, target):

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return torch.mean(torch.abs(pred_mag - target_mag))