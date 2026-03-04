"""Simulate federated training locally using Flower's simulation API."""
from dataset import TelemetryDatasetGenerator
from feature_pipeline import dataframe_to_feature_matrix, feature_dim
from client import create_flwr_client
import numpy as np
from typing import Dict, Any
from risk_engine import prob_to_score, make_alerts

# We'll import Flower inside run_simulation when needed
FLOWER_AVAILABLE = False


def _ensure_flower():
    global FLOWER_AVAILABLE, fl, get_strategy
    try:
        import flwr as fl
        from server import get_strategy
        FLOWER_AVAILABLE = True
    except ImportError:
        FLOWER_AVAILABLE = False


def make_clients(num_clients: int = 3, n_windows: int = 500) -> Dict[str, Dict[str, Any]]:
    gen = TelemetryDatasetGenerator()
    clients = {}
    for i in range(num_clients):
        node_id = f'node-{i+1}'
        df = gen.generate_node_data(node_id, n_windows=n_windows, anomaly_frac=0.03)
        X, y = dataframe_to_feature_matrix(df)
        clients[node_id] = {'X': X, 'y': y, 'timestamps': df['timestamp'].tolist()}
    return clients


def run_simulation(num_clients: int = 3, rounds: int = 3, dp_config: Dict = None, simple: bool = False, round_callback=None):
    # optionally load Flower APIs
    _ensure_flower()
    clients = make_clients(num_clients=num_clients)
    input_dim = feature_dim()

    def client_fn(cid: str):
        # Flower simulation uses numeric string ids; map to our node identifiers
        if cid.isdigit():
            # convert 0->node-1, 1->node-2, etc.
            idx = int(cid)
            node_id = f'node-{idx+1}'
        else:
            node_id = cid
        data = clients[node_id]
        return create_flwr_client(node_id, data['X'], data['y'], input_dim, device='cpu', dp_config=dp_config)

    # If Flower simulation is available use it, otherwise run manual loop
    if FLOWER_AVAILABLE and not simple:
        strategy = get_strategy()
        client_ids = list(clients.keys())
        print(f'Starting Flower simulation with clients: {client_ids}')
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(clients),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    else:
        print('Flower not available; running manual federated training')
        # instantiate client objects
        client_objs = {cid: create_flwr_client(cid, clients[cid]['X'], clients[cid]['y'], feature_dim(), device='cpu', use_flwr=not simple)
                       for cid in clients}
        # simple FedAvg with metrics
        global_params = None
        history = {'rounds': [], 'client_train_loss': {}, 'client_avg_score': {}, 'alerts': {}}
        for r in range(rounds):
            print(f'Round {r+1}/{rounds}')
            updates = []
            weights = []
            per_client_loss = {}
            per_client_scores = {}
            per_client_alerts = {}
            for cid, client in client_objs.items():
                if global_params is not None:
                    client.set_parameters(global_params)
                params, n, metrics = client.fit(global_params or client.get_parameters(), config={})
                updates.append(params)
                weights.append(n)
                per_client_loss[cid] = metrics.get('train_loss', None)

                # compute risk score over client's windows
                X = clients[cid]['X']
                import torch
                client.model.eval()
                with torch.no_grad():
                    probs = client.model(torch.from_numpy(X.astype('float32'))).cpu().numpy()
                scores = [prob_to_score(float(p)) for p in probs]
                avg_score = float(np.mean(scores))
                per_client_scores[cid] = avg_score
                alerts = make_alerts(clients[cid]['timestamps'], probs, threshold_score=50.0)
                per_client_alerts[cid] = alerts

            # compute weighted average
            total = sum(weights)
            new_params = []
            for param_list in zip(*updates):
                averaged = sum(p * w for p, w in zip(param_list, weights)) / total
                new_params.append(averaged)
            global_params = new_params

            # record round metrics
            history['rounds'].append(r+1)
            history['client_train_loss'][r+1] = per_client_loss
            history['client_avg_score'][r+1] = per_client_scores
            history['alerts'][r+1] = per_client_alerts

            # invoke callback for live updates (dashboard)
            if round_callback is not None:
                try:
                    round_callback(r+1, per_client_loss, per_client_scores, per_client_alerts)
                except Exception:
                    pass

        print('Manual training completed')
        return history


if __name__ == '__main__':
    run_simulation(num_clients=3, rounds=3)
