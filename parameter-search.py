import subprocess
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from matplotlib.widgets import Slider

sns.set_theme(style="darkgrid", palette="muted")


def objective(trial: optuna.Trial) -> float:
    exploration_rate = trial.suggest_float("exploration_rate", 0.1, 2000.0, log=True)

    # fmt: off
    cmd = [
        "target/release/twenty-fourty-eight-solver",
        "--mode", "eval",
        "--algorithm", "mcts",
        "--search-time", "0.001",
        "-n", "1",
        "-C", str(exploration_rate)
    ]
    # fmt: on

    result = subprocess.run(
        cmd,
        env={"RUST_LOG": "info"},
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def exponential_moving_average(data: list, alpha=0.1):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])

    return ema


def main():

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    def run_optuna_study():
        study.optimize(
            objective,
            n_trials=1000,
            n_jobs=10,
            show_progress_bar=True,
        )

        best_trial = study.best_trial
        print("Best score:", best_trial.value)
        print("Best hyperparameters:", best_trial.params)

    optuna_thread = threading.Thread(target=run_optuna_study)
    optuna_thread.start()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.set_title("Live Hyperparameter Optimization")

    # Choose a rad color for both lines.
    line_color = "#ff4b4b"

    ax_alpha = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="#444444")
    alpha_slider = Slider(ax_alpha, "Rate", 0, 5, valinit=3)

    while optuna_thread.is_alive():
        # fmt: off
        trials = [
            t 
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        # fmt: on

        trials.sort(key=lambda x: x.number)

        if not trials:
            continue

        xs = [t.number for t in trials]
        ys = [t.value for t in trials]

        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.plot(
            xs,
            ys,
            color=line_color,
            linestyle="-",
            linewidth=1,
            alpha=0.2,
            label="Trial Score",
        )

        # Calculate and plot the exponential moving average with full opacity.
        alpha_val = np.exp(-alpha_slider.val).item()
        ema_y = exponential_moving_average(ys, alpha=alpha_val)
        label = "Exponential Moving Average"
        ax.plot(xs, ema_y, color=line_color, linestyle="-", linewidth=2, label=label)

        # Display the best score so far on the top-left.
        best_score = max(ys)
        ax.text(
            0.02,
            0.95,
            f"Best Score: {best_score:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Live Hyperparameter Optimization")
        ax.legend()

        fig.canvas.draw_idle()
        plt.pause(0.1)

    optuna_thread.join()
    plt.ioff()


if __name__ == "__main__":
    main()
