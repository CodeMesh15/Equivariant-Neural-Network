import torch.nn as nn
from equivariant_nn.layers.gauge_layers import GaugeEquivariantConv

class ShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GaugeEquivariantConv(3, 16)
        self.conv2 = GaugeEquivariantConv(16, 32)
        self.fc = nn.Linear(32 * 28 * 28, 10)  # Adjust as needed for your shape resolution

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
