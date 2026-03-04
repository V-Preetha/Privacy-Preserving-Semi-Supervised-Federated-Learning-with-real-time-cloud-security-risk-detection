"""Simple Tkinter dashboard to display live federated simulation metrics.

This dashboard runs the manual `run_simulation(..., simple=True, round_callback=...)`
and updates plots and logs in real time.
"""
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from simulation import run_simulation


class DashboardApp:
    def __init__(self, root, clients=2, rounds=3):
        self.root = root
        self.root.title('PP-SSFL Dashboard')
        self.clients = clients
        self.rounds = rounds

        # Left: log
        self.log = ScrolledText(root, width=60, height=20)
        self.log.grid(row=0, column=0, rowspan=2)

        # Right-top: loss plot
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(5,3))
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=root)
        self.canvas_loss.get_tk_widget().grid(row=0, column=1)
        self.ax_loss.set_title('Training Loss per Round')
        self.ax_loss.set_xlabel('Round')
        self.ax_loss.set_ylabel('Loss')

        # Right-bottom: avg risk score plot
        self.fig_score, self.ax_score = plt.subplots(figsize=(5,3))
        self.canvas_score = FigureCanvasTkAgg(self.fig_score, master=root)
        self.canvas_score.get_tk_widget().grid(row=1, column=1)
        self.ax_score.set_title('Avg Risk Score per Round')
        self.ax_score.set_xlabel('Round')
        self.ax_score.set_ylabel('Score')

        self.loss_history = []
        self.score_history = []

    def log_line(self, text: str):
        self.log.insert(tk.END, text + '\n')
        self.log.see(tk.END)

    def update_round(self, round_num, per_client_loss, per_client_scores, per_client_alerts):
        # Log summary
        self.log_line(f'Round {round_num} summary:')
        avg_loss = None
        losses = [v for v in per_client_loss.values() if v is not None]
        if losses:
            avg_loss = sum(losses) / len(losses)
            self.loss_history.append(avg_loss)
        else:
            self.loss_history.append(0.0)

        avg_scores = sum(per_client_scores.values()) / len(per_client_scores)
        self.score_history.append(avg_scores)

        self.log_line(f'  Avg train loss: {avg_loss}')
        for cid, l in per_client_loss.items():
            self.log_line(f'  {cid}: loss={l}, avg_score={per_client_scores[cid]:.2f}, alerts={len(per_client_alerts[cid])}')

        # Update plots
        rounds = list(range(1, len(self.loss_history)+1))
        self.ax_loss.clear(); self.ax_loss.plot(rounds, self.loss_history, marker='o')
        self.ax_loss.set_title('Training Loss per Round'); self.ax_loss.set_xlabel('Round'); self.ax_loss.set_ylabel('Loss')
        self.canvas_loss.draw()

        self.ax_score.clear(); self.ax_score.plot(rounds, self.score_history, marker='o', color='orange')
        self.ax_score.set_title('Avg Risk Score per Round'); self.ax_score.set_xlabel('Round'); self.ax_score.set_ylabel('Score')
        self.canvas_score.draw()

    def start_simulation_thread(self):
        def target():
            run_simulation(num_clients=self.clients, rounds=self.rounds, simple=True, round_callback=self.update_round)
            self.log_line('Simulation finished')
        t = threading.Thread(target=target, daemon=True)
        t.start()


if __name__ == '__main__':
    root = tk.Tk()
    app = DashboardApp(root, clients=3, rounds=4)
    app.start_simulation_thread()
    root.mainloop()
