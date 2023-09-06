import torch.nn as nn
import torch

__all__ = ['SharedMLP']


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.GroupNorm
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.GroupNorm
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(8, oc),
                Swish(),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)
