import torch.nn as nn
import torch
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d

__all__ = ['PVConv', 'Attention', 'Swish', 'PVConvReLU']


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super(Attention, self).__init__()
        assert in_ch % num_groups == 0
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)


    def forward(self, x):
        B, C = x.shape[:2]
        h = x




        q = self.q(h).reshape(B,C,-1)
        k = self.k(h).reshape(B,C,-1)
        v = self.v(h).reshape(B,C,-1)

        qk = torch.matmul(q.permute(0, 2, 1), k) #* (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B,C,*x.shape[2:])

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))

        return x

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Swish()
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Attention(out_channels, 8) if attention else Swish()
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, temb = inputs

        voxel_features, voxel_coords = self.voxelization(features, coords)
        from IPython import embed
        embed(header='check size')

        # TODO: check size

        voxel_features = self.voxel_layers(voxel_features)

        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        
        fused_features = voxel_features + self.point_features(features)

        return fused_features, coords, temb



class PVConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False, leak=0.2,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leak, True)
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            Attention(out_channels, 8) if attention else nn.LeakyReLU(leak, True)
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb
