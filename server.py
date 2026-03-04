"""Federated server configuration and helper for PP-SSFL.

Uses Flower's default FedAvg with configurable parameters.
"""
import flwr as fl


def get_strategy():
    # Use default FedAvg strategy; could be extended with custom aggregation
    # create a FedAvg strategy without optional args that may vary by version
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,
    )
    return strategy


if __name__ == '__main__':
    print('Server helper')
