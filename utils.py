"""Utility helpers for parameter conversion and reproducibility."""
import numpy as np
import torch
from typing import List


def parameters_to_weights(params: List[np.ndarray]) -> List[torch.Tensor]:
    return [torch.from_numpy(p.copy()) for p in params]


def weights_to_parameters(weights: List[torch.Tensor]) -> List[np.ndarray]:
    return [w.detach().cpu().numpy().astype(np.float32) for w in weights]


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
