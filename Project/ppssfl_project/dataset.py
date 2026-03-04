"""Telemetry dataset simulation for PP-SSFL research prototype.

Generates synthetic telemetry per node: system logs counts, network stats,
API call sequences, CPU/memory usage, with a small fraction of labeled
anomalies.
"""
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class TelemetryDatasetGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def _sample_api_sequence(self, length=10, attack=False):
        apis = [
            'auth.login', 'auth.logout', 'data.read', 'data.write',
            'compute.start', 'compute.stop', 'storage.put', 'storage.get',
            'config.update', 'metrics.report'
        ]
        if attack:
            # attacks include suspicious repeated API calls and admin actions
            seq = np.random.choice(['auth.login', 'data.read', 'config.update', 'storage.get'],
                                   size=length, p=[0.4, 0.2, 0.2, 0.2])
        else:
            seq = np.random.choice(apis, size=length)
        return list(seq)

    def generate_node_data(self, node_id: str, n_windows: int = 1000,
                           window_seconds: int = 60, anomaly_frac: float = 0.02):
        """Generate a DataFrame of telemetry windows for a single node.

        Columns: timestamp, syslog_count, net_in_kb, net_out_kb,
        api_calls (list), cpu_pct, mem_pct, label (0 normal, 1 anomaly, NaN unlabeled)
        """
        rows = []
        now = datetime.utcnow()
        n_anomaly = max(1, int(n_windows * anomaly_frac))
        anomaly_indices = set(np.random.choice(n_windows, size=n_anomaly, replace=False))

        for i in range(n_windows):
            ts = now + timedelta(seconds=i * window_seconds)
            is_anomaly = i in anomaly_indices

            # normal behavior
            if not is_anomaly:
                syslog_count = int(np.random.poisson(5))
                net_in = max(0.0, np.random.normal(200.0, 50.0))  # KB
                net_out = max(0.0, np.random.normal(180.0, 45.0))
                cpu = float(np.clip(np.random.normal(35.0, 10.0), 0, 100))
                mem = float(np.clip(np.random.normal(55.0, 8.0), 0, 100))
                api_calls = self._sample_api_sequence(length=8, attack=False)
                label = np.nan  # Most windows unlabeled
            else:
                # anomalous window
                syslog_count = int(np.random.poisson(30))
                net_in = max(0.0, np.random.normal(2000.0, 300.0))
                net_out = max(0.0, np.random.normal(1500.0, 300.0))
                cpu = float(np.clip(np.random.normal(85.0, 7.0), 0, 100))
                mem = float(np.clip(np.random.normal(90.0, 5.0), 0, 100))
                api_calls = self._sample_api_sequence(length=8, attack=True)
                label = 1  # Labeled anomaly

            rows.append({
                'node_id': node_id,
                'timestamp': ts,
                'syslog_count': syslog_count,
                'net_in_kb': net_in,
                'net_out_kb': net_out,
                'api_calls': api_calls,
                'cpu_pct': cpu,
                'mem_pct': mem,
                'label': label,
            })

        df = pd.DataFrame(rows)
        return df


def _quick_demo():
    gen = TelemetryDatasetGenerator()
    df = gen.generate_node_data('node-1', n_windows=200, anomaly_frac=0.03)
    print(df.head())


if __name__ == '__main__':
    _quick_demo()
