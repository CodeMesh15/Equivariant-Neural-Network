import torch.nn as nn
from equivariant_nn.layers.gcnn_layers import GCNNLayer

class MNISTRotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GCNNLayer(1, 8)
        self.layer2 = GCNNLayer(8, 16)
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
