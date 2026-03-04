"""Semi-supervised training utilities: pseudo-labeling + supervised training."""
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def generate_pseudo_labels(model: torch.nn.Module, X_unlabeled: np.ndarray,
                           device: torch.device, threshold: float = 0.9,
                           batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-labels for unlabeled examples using confidence threshold.

    Returns arrays (indices, pseudo_labels)
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_unlabeled.astype('float32')))
    loader = DataLoader(ds, batch_size=batch_size)
    indices = []
    pseudo = []
    idx = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            probs = model(xb).detach().cpu().numpy()
            for p in probs:
                if p >= threshold:
                    indices.append(idx)
                    pseudo.append(1.0)
                elif p <= (1 - threshold):
                    indices.append(idx)
                    pseudo.append(0.0)
                idx += 1
    return np.array(indices, dtype=int), np.array(pseudo, dtype=float)


def local_semi_supervised_train(model: torch.nn.Module, optimizer, criterion,
                                X_labeled: np.ndarray, y_labeled: np.ndarray,
                                X_unlabeled: np.ndarray, device: torch.device,
                                pseudo_threshold: float = 0.9,
                                epochs: int = 3, batch_size: int = 32):
    """Combined training loop using pseudo-labels from unlabeled data."""
    model.to(device)
    # Create labeled loader
    ds_lab = TensorDataset(torch.from_numpy(X_labeled.astype('float32')),
                          torch.from_numpy(y_labeled.astype('float32')))
    loader_lab = DataLoader(ds_lab, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        # Generate pseudo-labels each epoch
        if len(X_unlabeled) > 0:
            idxs, pseudos = generate_pseudo_labels(model, X_unlabeled, device,
                                                  threshold=pseudo_threshold,
                                                  batch_size=batch_size)
        else:
            idxs = np.array([], dtype=int); pseudos = np.array([], dtype=float)

        # Build combined dataset
        if len(idxs) > 0:
            X_pseudo = X_unlabeled[idxs]
            y_pseudo = pseudos
            X_comb = np.vstack([X_labeled, X_pseudo])
            y_comb = np.concatenate([y_labeled, y_pseudo])
        else:
            X_comb = X_labeled
            y_comb = y_labeled

        ds_comb = TensorDataset(torch.from_numpy(X_comb.astype('float32')),
                               torch.from_numpy(y_comb.astype('float32')))
        loader = DataLoader(ds_comb, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)

        avg_loss = total_loss / len(loader.dataset)
        # For simplicity we return last epoch loss
    return avg_loss


if __name__ == '__main__':
    import numpy as np
    from model import build_model
    model = build_model(16)
    Xl = np.random.randn(20, 16).astype('float32')
    yl = np.random.randint(0,2,size=(20,)).astype('float32')
    Xu = np.random.randn(100, 16).astype('float32')
    import torch.optim as optim
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss = local_semi_supervised_train(model, opt, F.binary_cross_entropy,
                                       Xl, yl, Xu, torch.device('cpu'))
    print('Done, loss=', loss)
