import torch
import numpy as np


def rotate_tensor(x, angle, axis=2):
    """Rotate a 2D or 3D tensor around the specified axis."""
    theta = angle * np.pi / 180
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = torch.tensor([
        [c, -s],
        [s,  c]
    ], dtype=torch.float32)
    if x.ndim == 2:
        return x @ rotation_matrix.T
    elif x.ndim == 3:
        return torch.einsum('bij,jk->bik', x, rotation_matrix)
    else:
        raise ValueError("Input must be 2D or 3D tensor")


def reflection_tensor(x):
    """Reflect tensor along the x-axis (simple example)."""
    reflection_matrix = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
    if x.ndim == 2:
        return x @ reflection_matrix.T
    elif x.ndim == 3:
        return torch.einsum('bij,jk->bik', x, reflection_matrix)
    else:
        raise ValueError("Input must be 2D or 3D tensor")
