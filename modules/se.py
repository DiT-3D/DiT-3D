import torch.nn as nn
import torch
__all__ = ['SE3d']

class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)
class SE3d(nn.Module):
    def __init__(self, channel, reduction=8, use_relu=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(True) if use_relu else Swish() ,
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)
