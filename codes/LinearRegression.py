import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.Linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.Linear(x)
        return out
