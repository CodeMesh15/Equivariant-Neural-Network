import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.nn import Gate
from e3nn.o3 import Linear

class SE3EquivariantLayer(nn.Module):
    """
    SE(3)-equivariant layer using e3nn library.
    """

    def __init__(self, input_irreps='1x0e', output_irreps='1x0e + 1x1e'):
        super().__init__()
        self.tp = Linear(Irreps(input_irreps), Irreps(output_irreps))

    def forward(self, x):
        return self.tp(x)
