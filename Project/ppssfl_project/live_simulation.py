"""Utilities to initialize a demo model and run inference on single records."""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import build_model
from semi_supervised_training import local_semi_supervised_train
from telemetry_generator import RealtimeTelemetryGenerator


def telemetry_to_features(api_calls, cpu_usage, memory_usage, network_traffic):
    """Convert telemetry metrics to a feature vector.
    
    Normalizes each metric to 0-100 range for consistency.
    
    Returns:
        np.ndarray: feature vector of shape (4,)
    """
    # Normalize api_calls to 0-100 (max is ~200)
    api_norm = min(float(api_calls) / 2.0, 100.0)
    
    # CPU and Memory are already 0-100
    cpu_norm = float(cpu_usage)
    mem_norm = float(memory_usage)
    
    # Normalize network traffic to 0-100 (max is ~1200)
    net_norm = min(float(network_traffic) / 12.0, 100.0)
    
    return np.array([api_norm, cpu_norm, mem_norm, net_norm], dtype=np.float32)


def init_model_for_demo(hidden_dim: int = 64, n_samples: int = 300, seed: int = 42):
    """Initialize and lightly train a model on synthetic telemetry for demo.

    Args:
        hidden_dim: Hidden layer dimension
        n_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility

    Returns:
        Trained PyTorch model on CPU.
    """
    # Generate synthetic telemetry data
    gen = RealtimeTelemetryGenerator(anomaly_prob=0.07, use_seed=True)
    np.random.seed(seed)
    
    X_all = []
    y_all = []
    
    for _ in range(n_samples):
        record = gen.next_record()
        
        # Convert to features
        features = telemetry_to_features(
            record['api_calls'],
            record['cpu_usage'],
            record['memory_usage'],
            record['network_traffic']
        )
        X_all.append(features)
        
        # Label: 1 if anomaly, 0 if normal
        label = 1 if record['event_type'] == 'anomaly' else 0
        y_all.append(label)
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    
    input_dim = X_all.shape[1]  # 4 features
    model = build_model(input_dim, hidden_dim=hidden_dim)
    device = torch.device('cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Build labeled/unlabeled split
    pos_mask = (y_all == 1)
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]
    
    # Ensure we have both classes
    n_pos = min(len(pos_idx), 50)  # Cap positive samples at 50
    n_neg = min(len(neg_idx), 100)  # Cap negative samples at 100
    
    if n_pos > 0 and n_neg > 0:
        sampled_pos = np.random.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        sampled_neg = np.random.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        
        X_labeled = np.vstack([X_all[sampled_pos], X_all[sampled_neg]])
        y_labeled = np.concatenate([np.ones(len(sampled_pos)), np.zeros(len(sampled_neg))]).astype(float)
        
        # Remaining samples are unlabeled
        used_idx = np.concatenate([sampled_pos, sampled_neg])
        unused_idx = np.setdiff1d(np.arange(len(X_all)), used_idx)
        X_unlabeled = X_all[unused_idx] if len(unused_idx) > 0 else X_all[:0]
    else:
        # Fallback if no anomalies present
        X_labeled = X_all[:100]
        y_labeled = y_all[:100].astype(float)
        X_unlabeled = X_all[100:] if len(X_all) > 100 else X_all[:0]
    
    # Train model with semi-supervised approach
    model.train()
    local_semi_supervised_train(
        model, optimizer, F.binary_cross_entropy,
        X_labeled, y_labeled, X_unlabeled, device,
        pseudo_threshold=0.9, epochs=5, batch_size=32
    )
    
    return model


def infer_record(model: torch.nn.Module, record: dict) -> float:
    """Run inference on a single telemetry record and return probability.

    Args:
        model: PyTorch model
        record: dict produced by RealtimeTelemetryGenerator

    Returns:
        float: Probability (0-1) that record is an anomaly
    """
    features = telemetry_to_features(
        record['api_calls'],
        record['cpu_usage'],
        record['memory_usage'],
        record['network_traffic']
    )
    
    x = torch.from_numpy(features.astype('float32')).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        p = model(x).cpu().numpy().flatten()[0]
    
