"""Differential privacy training utilities using Opacus.

Provides a helper to attach Opacus PrivacyEngine to an optimizer and
wrap training loops for DP-SGD. If Opacus is unavailable, falls back to
non-DP training with a warning.
"""
from typing import Optional, Dict
import warnings
import torch

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except Exception:
    OPACUS_AVAILABLE = False


def make_private(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 sample_rate: float, noise_multiplier: float,
                 max_grad_norm: float, epochs: int):
    """Attach Opacus PrivacyEngine to optimizer and return it.

    If Opacus is unavailable, return None.
    """
    if not OPACUS_AVAILABLE:
        warnings.warn('Opacus not available — running without differential privacy')
        return None

    privacy_engine = PrivacyEngine()
    privacy_engine.attach(
        module=model,
        optimizer=optimizer,
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine


def get_privacy_summary(privacy_engine) -> Optional[Dict]:
    if privacy_engine is None:
        return None
    try:
        return {
            'epsilon': privacy_engine.get_epsilon(delta=1e-5),
            'delta': 1e-5
        }
    except Exception:
        return None


if __name__ == '__main__':
    print('Opacus available:', OPACUS_AVAILABLE)
