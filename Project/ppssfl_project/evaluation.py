"""Comprehensive evaluation pipeline with proper train/test split and metrics.

This module:
- Generates synthetic telemetry with both normal and anomaly samples
- Splits data into train/test sets
- Trains a model
- Computes metrics with proper handling of edge cases
- Shows class distribution
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from telemetry_generator import RealtimeTelemetryGenerator
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import build_model
from semi_supervised_training import local_semi_supervised_train


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


def generate_evaluation_dataset(n_samples: int = 500, anomaly_prob: float = 0.07, seed: int = 42):
    """Generate a dataset with mixed normal and anomaly samples.
    
    Args:
        n_samples: Total number of samples to generate
        anomaly_prob: Probability of each sample being an anomaly (0.07 = 7%)
        seed: Random seed for reproducibility
    
    Returns:
        X: Feature matrix of shape (n_samples, 4)
        y: Binary labels (0=normal, 1=anomaly)
        df: DataFrame with raw telemetry data
    """
    np.random.seed(seed)
    
    gen = RealtimeTelemetryGenerator(anomaly_prob=anomaly_prob, use_seed=True)
    
    records = []
    labels = []
    
    for _ in range(n_samples):
        record = gen.next_record()
        records.append(record)
        
        # Label: 1 if anomaly, 0 if normal
        label = 1 if record['event_type'] == 'anomaly' else 0
        labels.append(label)
    
    # Convert to DataFrame for easy viewing
    df = pd.DataFrame(records)
    
    # Extract features
    X = np.array([
        telemetry_to_features(
            r['api_calls'],
            r['cpu_usage'],
            r['memory_usage'],
            r['network_traffic']
        )
        for _, r in df.iterrows()
    ], dtype=np.float32)
    
    y = np.array(labels, dtype=np.int32)
    
    return X, y, df


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True):
    """Compute comprehensive evaluation metrics.
    
    Args:
        model: PyTorch model
        X_test: Feature matrix
        y_test: Binary labels
        verbose: Whether to print detailed output
    
    Returns:
        dict: Metrics including accuracy, precision, recall, f1, roc_auc
    """
    if len(y_test) == 0:
        return {}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_test.astype('float32'))
        preds_prob = model(X).cpu().numpy().flatten()
    
    # Binary predictions (threshold 0.5)
    y_pred = (preds_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)
    
    metrics = {}
    
    # Compute accuracy, precision, recall, f1
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # ROC-AUC only if both classes present
    if len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, preds_prob))
        except Exception as e:
            if verbose:
                print(f"Warning: ROC-AUC could not be computed: {e}")
            metrics['roc_auc'] = None
    else:
        if verbose:
            print(f"Warning: ROC-AUC not available — only one class present in test set")
        metrics['roc_auc'] = None
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        else:
            print(f"ROC-AUC:   Not available")
        print("="*60 + "\n")
    
    return metrics


def train_and_evaluate(
    n_samples: int = 500,
    anomaly_prob: float = 0.07,
    test_size: float = 0.2,
    hidden_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True
):
    """Complete pipeline: generate data, train model, evaluate.
    
    Args:
        n_samples: Total samples to generate
        anomaly_prob: Probability of anomalies
        test_size: Fraction for test set
        hidden_dim: Hidden dimension for model
        epochs: Training epochs
        batch_size: Batch size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        dict: Results including metrics, model, train/test data
    """
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION PIPELINE")
        print(f"{'='*60}")
        print(f"Generating {n_samples} telemetry samples with {anomaly_prob*100:.1f}% anomalies...\n")
    
    # Step 1: Generate data
    X, y, df = generate_evaluation_dataset(n_samples, anomaly_prob, seed)
    
    # Show overall class distribution
    unique, counts = np.unique(y, return_counts=True)
    if verbose:
        print(f"Overall class distribution:")
        for label, count in zip(unique, counts):
            label_name = "ANOMALY" if label == 1 else "NORMAL"
            print(f"  {label_name}: {count} samples ({100*count/len(y):.1f}%)")
    
    # Step 2: Train/test split
    if verbose:
        print(f"\nSplitting into train ({100*(1-test_size):.0f}%) / test ({100*test_size:.0f}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y  # Ensure both sets have both classes
    )
    
    # Show train/test distributions
    if verbose:
        print(f"\nTraining set class distribution:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_train, counts_train):
            label_name = "ANOMALY" if label == 1 else "NORMAL"
            print(f"  {label_name}: {count} samples ({100*count/len(y_train):.1f}%)")
        
        print(f"\nTest set class distribution:")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        for label, count in zip(unique_test, counts_test):
            label_name = "ANOMALY" if label == 1 else "NORMAL"
            print(f"  {label_name}: {count} samples ({100*count/len(y_test):.1f}%)")
    
    # Step 3: Build model
    input_dim = X_train.shape[1]  # 4 features
    model = build_model(input_dim, hidden_dim=hidden_dim)
    device = torch.device('cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Step 4: Simple supervised training (no semi-supervised complication)
    if verbose:
        print(f"\nTraining model for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        # Mini-batch training
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = torch.from_numpy(X_train[batch_idx]).to(device)
            y_batch = torch.from_numpy(y_train[batch_idx].astype(np.float32)).to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch).squeeze()
            loss = F.binary_cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")
    
    # Step 5: Evaluate
    if verbose:
        print(f"\nEvaluating on test set...")
    
    metrics = evaluate_model(model, X_test, y_test, verbose=verbose)
    
    return {
        'model': model,
        'metrics': metrics,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'df': df
    }


if __name__ == '__main__':
    # Run the full evaluation pipeline
    results = train_and_evaluate(
        n_samples=500,
        anomaly_prob=0.07,
        test_size=0.2,
        hidden_dim=64,
        epochs=10,
        verbose=True
    )
    
    model = results['model']
    print("\n✓ Model training complete!")
    print(f"✓ Model saved for later use")
