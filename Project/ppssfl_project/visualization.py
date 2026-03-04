"""Visualization utilities for training metrics and risk trends."""
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_losses(rounds: List[int], losses: List[float], outpath: str = None):
    plt.figure()
    plt.plot(rounds, losses, marker='o')
    plt.xlabel('Federated Round')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Federated Rounds')
    plt.grid(True)
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()


def plot_risk_trends(timestamps: List, scores: List[float], outpath: str = None):
    plt.figure()
    plt.plot(timestamps, scores, marker='-')
    plt.xlabel('Time')
    plt.ylabel('Risk Score (0-100)')
    plt.title('Risk Score Trend')
    plt.grid(True)
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()


def plot_alert_timeline(alerts: List[Dict], outpath: str = None):
    if len(alerts) == 0:
        print('No alerts to plot')
        return
    timestamps = [a['timestamp'] for a in alerts]
    scores = [a['score'] for a in alerts]
    plt.figure(figsize=(10, 2))
    plt.scatter(timestamps, scores, c='red')
    plt.yticks([25,50,75,100])
    plt.xlabel('Time')
    plt.title('Anomaly Alert Timeline')
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()


if __name__ == '__main__':
    print('Visualization utilities')
