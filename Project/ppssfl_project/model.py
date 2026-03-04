"""PyTorch model for risk prediction.

Simple feed-forward encoder + hidden layers + probability head.
"""
import torch
import torch.nn as nn


class RiskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.hidden(x)
        p = self.head(x).squeeze(-1)
        return p


def build_model(input_dim: int, hidden_dim: int = 64):
    return RiskModel(input_dim=input_dim, hidden_dim=hidden_dim)


if __name__ == '__main__':
    m = build_model(16)
    x = torch.randn(4, 16)
    print(m(x))
