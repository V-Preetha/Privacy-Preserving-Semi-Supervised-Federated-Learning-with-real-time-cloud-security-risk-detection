"""Feature engineering pipeline for telemetry windows.

Converts telemetry DataFrame rows into numerical feature vectors suitable
for PyTorch models.
"""
from collections import Counter
import numpy as np
import pandas as pd
from typing import List, Tuple


API_VOCAB = [
    'auth.login', 'auth.logout', 'data.read', 'data.write',
    'compute.start', 'compute.stop', 'storage.put', 'storage.get',
    'config.update', 'metrics.report'
]


def encode_api_sequence(seq: List[str]) -> np.ndarray:
    """Simple bag-of-apis encoding (counts normalized)."""
    c = Counter(seq)
    vec = np.array([c[a] for a in API_VOCAB], dtype=np.float32)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec


def row_to_features(row: pd.Series) -> np.ndarray:
    """Convert a telemetry row to a numeric feature vector."""
    api_vec = encode_api_sequence(row['api_calls'])
    net_in = float(row['net_in_kb'])
    net_out = float(row['net_out_kb'])
    syslog = float(row['syslog_count'])
    cpu = float(row['cpu_pct'])
    mem = float(row['mem_pct'])

    stat_features = np.array([
        net_in, net_out, net_in - net_out, syslog, cpu, mem
    ], dtype=np.float32)

    feat = np.concatenate([api_vec, stat_features])
    return feat


def dataframe_to_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to feature matrix X and label vector y (NaN labels remain).

    Returns:
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with values {0,1,np.nan}
    """
    X = np.stack([row_to_features(r) for _, r in df.iterrows()])
    y = df['label'].to_numpy(dtype=float)
    return X, y


def feature_dim() -> int:
    return len(API_VOCAB) + 6


if __name__ == '__main__':
    import pandas as pd
    from dataset import TelemetryDatasetGenerator

    gen = TelemetryDatasetGenerator()
    df = gen.generate_node_data('demo', n_windows=10)
    X, y = dataframe_to_feature_matrix(df)
    print('X.shape=', X.shape, 'y.shape=', y.shape)
