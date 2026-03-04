"""Entry point to run PP-SSFL federated simulation locally."""
from simulation import run_simulation
from visualization import plot_losses
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--simple', action='store_true', help='Use manual federated loop instead of Flower')
    args = parser.parse_args()

    # For this prototype we run a small simulation
    run_simulation(num_clients=args.clients, rounds=args.rounds, simple=args.simple)
    if args.simple:
        print('Simple mode requested; manual loops executed (Flower simulation bypassed)')


if __name__ == '__main__':
    main()
