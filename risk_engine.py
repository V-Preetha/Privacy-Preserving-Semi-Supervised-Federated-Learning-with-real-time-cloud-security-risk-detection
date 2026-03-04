"""Risk scoring engine: maps probabilities to 0-100 and risk bands, alerts."""
from typing import Tuple, Dict


def prob_to_score(p: float) -> float:
    """Convert probability [0,1] to score [0,100]."""
    return float(max(0.0, min(1.0, p))) * 100.0


def score_to_band(score: float) -> Tuple[str, int]:
    if score <= 25:
        return 'Normal', 0
    elif score <= 50:
        return 'Elevated', 1
    elif score <= 75:
        return 'High', 2
    else:
        return 'Critical', 3


def make_alerts(timestamps, probs, threshold_score: float = 50.0):
    """Return alerts for timestamps where score exceeds threshold."""
    alerts = []
    for ts, p in zip(timestamps, probs):
        s = prob_to_score(p)
        if s >= threshold_score:
            band, _ = score_to_band(s)
            alerts.append({'timestamp': ts, 'score': s, 'band': band})
    return alerts


if __name__ == '__main__':
    print('Risk engine loaded')
