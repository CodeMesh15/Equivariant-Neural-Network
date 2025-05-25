import torch
import torch.nn as nn

class GaugeEquivariantConv(nn.Module):
    """
    Simplified gauge equivariant convolution layer.
    Typically used for functions on manifolds with local frames.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x, connection=None):
        # 'connection' can be used for parallel transport — not implemented here
        return self.kernel(x)
