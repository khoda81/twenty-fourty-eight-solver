import copy
import json
import os
import pickle
import subprocess
import sys
import threading
from multiprocessing import Queue, cpu_count, pool

import numpy as np
from matplotlib import pyplot as plt
from skopt import Optimizer, space, utils
from tqdm import tqdm


def format_params(params: list, param_space: list[space.Dimension]) -> str:
    cmd = []
    for k, v in zip(param_space, params):
        if isinstance(v, (bool, np.bool_)):
            if v:
                cmd.append(f"{k.name}")

        elif isinstance(v, float):
            cmd.extend([f"{k.name}", f"{v:.2f}"])

        else:
            cmd.extend([f"{k.name}", f"{v}"])

    return " ".join(cmd)


def evaluate_parameters(command: list[str]):
    """Run the command with given parameters and return the score."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception automatically
            timeout=240,  # Avoid infinite hangs
            env={**os.environ, "RUST_LOG": "info"},
        )

        if result.returncode != 0:
            print(
                f"Subprocess failed with code {result.returncode}: {result.stderr}",
                file=sys.stderr,
            )
            return 0.0  # Penalize failures

        output = result.stdout.strip()
        return -float(output)

    except subprocess.CalledProcessError as e:
        print(f"Error running {command}: {e.stderr}", file=sys.stderr)
        return 0.0

    except ValueError:
        print(f"Invalid output for {command}: {result.stdout}", file=sys.stderr)
        return 0.0


def convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


def update_plot(
    ax: plt.Axes,
    results: utils.OptimizeResult,
    current_jobs: dict[int, list],
    param_space: list[space.Dimension],
):
    ax.clear()

    # Extract data from optimization results
    C_values = [x[0] for x in results.x_iters]
    cache_flags = [x[1] for x in results.x_iters]
    scores = [-score for score in results.func_vals]

    # Plot results
    for cache_val in [False, True]:
        mask = [flag == cache_val for flag in cache_flags]
        color = "red" if cache_val else "blue"
        ax.scatter(
            np.array(C_values)[mask],
            np.array(scores)[mask],
            label=f"Cache: {cache_val}",
            color=color,
            alpha=0.6,
        )

    # Plot currently running parameters
    for params in current_jobs.values():
        color = "red" if params[1] else "blue"
        ax.axvline(params[0], color=color, linestyle="dashed", alpha=0.7)

    ax.set_xscale("log")
    ax.set_title("Parameter Performance")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Score")
    ax.legend()
    plt.draw()
    plt.pause(0.1)


def search(n_iter=300, n_jobs=cpu_count() - 1):
    """Run asynchronous Bayesian optimization with state loading and visualization."""
    base_cmd = [
        r"target\release\twenty-fourty-eight-solver.exe",
        "--mode=eval",
        "--search-time=0.05",
        "--num-eval-games=1",
    ]

    param_space: list[space.Dimension] = [
        space.Real(0.1, 2000, name="-C", prior="log-uniform"),
        space.Categorical([False, True], name="--persistent-cache"),
    ]

    save_file = "optimizer_state.pckl"
    optimizer = Optimizer(param_space, n_initial_points=n_jobs)
    initial_completed = 0

    # Load previous state if available
    if os.path.exists(save_file):
        print(f"Loading optimizer from {save_file}")
        with open(save_file, "rb") as f:
            optimizer = pickle.loads(open(save_file, "rb").read())

    result_queue = Queue()
    thread_pool = pool.ThreadPool(n_jobs)
    job_lock = threading.Lock()
    current_jobs = {}
    completed = 0
    pbar = tqdm(
        total=n_iter,
        initial=initial_completed,
        smoothing=0,
    )

    def submit_job(params: list):
        nonlocal current_jobs, completed
        job_id = completed + len(current_jobs)
        # print(f"Launching job[{job_id}]: {params}, {current_jobs}")
        with job_lock:
            current_jobs[job_id] = copy.deepcopy(params)
        # print(f"Launching job[{job_id}]: {params}, {current_jobs}")

        def callback(score):
            nonlocal completed
            completed += 1
            with job_lock:
                result_queue.put((current_jobs.pop(job_id), score))

        def error_callback(e):
            print(f"Job failed: {e}")
            with job_lock:
                current_jobs.pop(job_id)

        cmd = get_command(current_jobs[job_id])

        thread_pool.apply_async(
            evaluate_parameters,
            (cmd,),
            callback=callback,
            error_callback=error_callback,
        )

    def get_command(params: list):
        cmd = base_cmd.copy()
        for k, v in zip(param_space, params):
            if isinstance(v, (bool, np.bool_)):
                if v:
                    cmd.append(f"{k.name}")
            else:
                cmd.extend([f"{k.name}", str(v)])

        return cmd

    # Submit initial batch
    pbar.set_description("Submitting initial batch")
    initial_batch = min(n_jobs, n_iter)
    for params in optimizer.ask(n_points=initial_batch):
        submit_job(params)

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        while completed < n_iter:
            pbar.set_description("Waiting")
            while result_queue.empty():
                plt.pause(0.01)

            params, score = result_queue.get()
            pbar.update(1)

            while not result_queue.empty():
                params, score = result_queue.get_nowait()
                pbar.update(1)
                optimizer.tell(params, score, fit=False)

            pbar.set_description("Fitting")
            result = optimizer.tell(params, score)

            # Submit new jobs
            with job_lock:
                idle = n_jobs - len(current_jobs)

            if idle:
                remaining = n_iter - completed
                new_jobs = min(idle, remaining)

                pbar.set_description(f"Selecting {new_jobs}")
                params = optimizer.ask(n_points=new_jobs)

                pbar.set_description("Submitting")
                for params in params:
                    submit_job(params)

            # Update visualization
            with job_lock:
                pbar.set_description("Updating plot")
                update_plot(ax, result, current_jobs, param_space)

                pbar.set_description("Getting best")
                best_params = optimizer.ask(strategy="best")  # Get best expected para
                pbar.set_postfix(
                    {
                        "best_params": format_params(best_params, param_space),
                        "jobs": list(current_jobs.keys()),
                    }
                )

    finally:
        thread_pool.close()
        thread_pool.join()
        pbar.close()

        with open(save_file, "wb") as f:
            pickle.dump(optimizer, f)

        print(f"\nOptimization state saved to {save_file}")

        plt.ioff()

        best_params = optimizer.ask(strategy="best")  # Get best expected params
        print(f"\nBest parameters: {get_command(best_params)}")
        if optimizer.models:
            best_score_mean, best_score_std = optimizer.models[-1].predict(
                [best_params], return_std=True
            )

            print(
                f"Expected average score: {-best_score_mean[0]:.2f} ± {best_score_std[0]:.2f}"
            )  # Mean ± StdDev


if __name__ == "__main__":
    search(n_iter=300)
