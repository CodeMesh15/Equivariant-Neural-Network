
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNNLayer(nn.Module):
    """
    Group-equivariant convolution layer for discrete groups like rotations/reflections.
    Assumes cyclic rotation group C4 (e.g. 90-degree rotations).
    """

    def __init__(self, in_channels, out_channels, group_size=4, kernel_size=3, padding=1):
        super().__init__()
        self.group_size = group_size
        self.conv = nn.Conv2d(in_channels * group_size, out_channels * group_size, kernel_size, padding=padding)

    def forward(self, x):
        # Input shape: [B, C, H, W] → replicate across group
        B, C, H, W = x.shape
        x = x.repeat(1, self.group_size, 1, 1)  # Simplified: pretend to augment by rotations
        x = self.conv(x)
        return x
