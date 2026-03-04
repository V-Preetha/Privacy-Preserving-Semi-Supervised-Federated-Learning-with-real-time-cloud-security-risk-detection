"""Flower NumPy client implementation for PP-SSFL prototype."""
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch import nn
from model import build_model
from semi_supervised_training import local_semi_supervised_train
from dp_training import make_private, get_privacy_summary
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

# We'll conditionally import Flower when needed to avoid heavy dependencies
fl = None


class BaseClient:
    """Simple client implementation without Flower dependencies."""

    def __init__(self, cid: str, X: np.ndarray, y: np.ndarray,
                 input_dim: int, device: str = 'cpu', dp_config: Optional[dict] = None):
        self.cid = cid
        self.device = torch.device(device)
        self.model = build_model(input_dim)
        self.dp_config = dp_config or {}
        labeled_mask = ~np.isnan(y)
        self.X_labeled = X[labeled_mask]
        self.y_labeled = y[labeled_mask].astype('float32') if self.X_labeled.size else np.empty((0,))
        self.X_unlabeled = X[~labeled_mask]

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        keys = list(self.model.state_dict().keys())
        params = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = float(config.get('lr', 1e-3))
        epochs = int(config.get('local_epochs', 1))
        device = self.device
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if self.dp_config:
            make_private(self.model, optimizer,
                         sample_rate=self.dp_config.get('sample_rate', 0.01),
                         noise_multiplier=self.dp_config.get('noise_multiplier', 1.0),
                         max_grad_norm=self.dp_config.get('max_grad_norm', 1.0),
                         epochs=epochs)
        import numpy as _np
        if self.X_labeled.shape[0] == 0:
            n = min(10, max(1, int(0.05 * len(self.X_unlabeled))))
            Xl = self.X_unlabeled[:n]
            yl = _np.zeros((n,), dtype='float32')
        else:
            Xl = self.X_labeled; yl = self.y_labeled
        loss = local_semi_supervised_train(self.model, optimizer, F.binary_cross_entropy,
                                           Xl, yl, self.X_unlabeled, device,
                                           pseudo_threshold=float(config.get('pseudo_threshold', 0.9)),
                                           epochs=epochs, batch_size=int(config.get('batch_size', 32)))
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # return params, number of examples, and metrics including training loss
        return params, len(Xl) + len(self.X_unlabeled), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.X_labeled.shape[0] == 0:
            return float(0.0), 0, {}
        self.model.eval()
        X = torch.from_numpy(self.X_labeled.astype('float32'))
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        y = self.y_labeled
        eps = 1e-7
        loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
        acc = float((preds >= 0.5).astype(int).flatten() == y).mean()
        return float(loss), int(len(y)), {"accuracy": float(acc)}

# FLClient class will be generated dynamically when Flower is requested


def create_flwr_client(cid: str, X: np.ndarray, y: np.ndarray, input_dim: int, device='cpu', dp_config=None, use_flwr: bool = True):
    """Factory returning either a Flower NumPy client or plain BaseClient."""
    global fl
    if use_flwr and fl is None:
        try:
            import flwr as _fl
            fl = _fl
        except ImportError:
            fl = None
    if use_flwr and fl is not None:
        # dynamically construct Flower-compatible client class
        class FLClientLocal(BaseClient, fl.client.NumPyClient):
            def __init__(self, *args, **kwargs):
                BaseClient.__init__(self, *args, **kwargs)
        client = FLClientLocal(cid, X, y, input_dim, device=device, dp_config=dp_config)
        # some Flower versions require converting
        if hasattr(client, 'to_client'):
            return client.to_client()
        return client
    else:
        return BaseClient(cid, X, y, input_dim, device=device, dp_config=dp_config)
