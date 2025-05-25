import torch.nn as nn
from equivariant_nn.layers.se3_layers import SE3EquivariantLayer

class QM9Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.se3_1 = SE3EquivariantLayer('1x0e', '1x1e + 1x0e')
        self.se3_2 = SE3EquivariantLayer('1x1e + 1x0e', '1x0e')
        self.fc = nn.Linear(1, 1)  # Output property (e.g., molecular energy)

    def forward(self, x, pos, edge_index):
        # x: node features, pos: coordinates, edge_index: connectivity
        h = self.se3_1(x)
        h = self.se3_2(h)
        out = self.fc(h.mean(dim=1))  # Global pooling
        return out
