import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from typing import Callable
import warnings
from collections import defaultdict

# Local imports
from common import metrics, plot
from common.filters import moving_avg, kalman


def main(N: int = 50,
         filter: Callable = moving_avg,
         save_plot: bool = True,
         save_metrics: bool = True,
         max_ep: int = int(3e6 / 150)) -> None:
    """
    Perform full analysis on the results of the training runs.
    """
    algos = ["A2C", "PPO", "SAC", "TD3", "DDPG"]
    training = np.linspace(0, 100, 6)  # [0, 20, 40, 60, 80, 100] %

    base_dir = "experiments/"
    seed_dirs = sorted(d for d in os.listdir(base_dir) if d.startswith("ATC_RL."))
    if not seed_dirs:
        raise RuntimeError("No seed folders found in 'experiments/'. Expected 'ATC_RL.<seed>'.")

    print(f"Found {len(seed_dirs)} seed folders in '{base_dir}': {seed_dirs}")
    seeds = [int(d.split(".")[1]) for d in seed_dirs]
    print(f"Extracted seeds: {seeds}")

    # store list of reward-series per (algo, pct)
    results = {algo: defaultdict(list) for algo in algos}
    times = {algo: [] for algo in algos}

    for exp in seed_dirs:
        exp_dir = os.path.join(base_dir, exp)
        print(f"Processing experiment directory: {exp_dir}")

        for algo in algos:
            # extract the timestamps for the current algo
            time_path = os.path.join(exp_dir, f"{algo}_ts.csv")
            
            # load the timestamps
            if os.path.exists(time_path):
                time_df = pd.read_csv(time_path)
                times[algo].append(time_df)
            else:
                warnings.warn(f"Missing timestamps for {algo} in {exp_dir}.")
                continue
            
            for pct in training:
                # assume subfolders "lofi_{algo}/<algo>_{pct}/logs/results.csv"
                lofi_path = os.path.join(
                    exp_dir,
                    f"LoFi-{algo}",
                    f"{algo}_{pct:.1f}",
                    "logs",
                    "results.csv",
                )
                hifi_path = os.path.join(
                    exp_dir,
                    f"HiFi-{algo}",
                    f"{algo}_{pct:.1f}",
                    "logs",
                    "results.csv",
                )
                print(f"Loading results for {algo} at {pct:.1f}% from {lofi_path} and {hifi_path}")

                if not os.path.exists(lofi_path) or not os.path.exists(hifi_path):
                    warnings.warn(
                        f"Missing results files for {algo} at {pct:.1f}% in {exp_dir}."
                    )
                    continue

                lofi_df = pd.read_csv(lofi_path)
                hifi_df = pd.read_csv(hifi_path)

                # drop unwanted columns
                for df in (lofi_df, hifi_df):
                    if "timestep" in df.columns:
                        df.drop(columns=["timestep"], inplace=True)

                # trim to the right number of episodes
                lo_cut = int(pct * max_ep / 100)+1
                hi_cut = int((100 - pct) * max_ep / 100)
                lofi_df = lofi_df.iloc[:lo_cut]
                hifi_df = hifi_df.iloc[:hi_cut]

                # stitch episodes together
                if not lofi_df.empty:
                    offset = lofi_df["episode"].max() + 1
                else:
                    offset = 0
                # use .loc to avoid SettingWithCopyWarning
                hifi_df.loc[:, "episode"] = hifi_df["episode"] + offset

                # now stitch episodes together
                exp_df = pd.concat([lofi_df, hifi_df], ignore_index=True)
                # … then your existing cast …
                exp_df = exp_df.astype({"episode": int})

                if "reward" not in exp_df.columns:
                    raise RuntimeError("Column 'reward' not found in results.csv")

                # index by episode, store the reward series
                rewards = exp_df.set_index("episode")["reward"]
                results[algo][pct].append(rewards)

    # aggregate across seeds
    aggregated = {algo: {} for algo in algos}
    avg_times = {algo: {} for algo in algos}
    for algo in algos:
        if times[algo]:
            # concatenate all seed DataFrames
            time_df_all = pd.concat(times[algo], ignore_index=True)
            # group by percentage
            grp = time_df_all.groupby('percentage')[['lofi-duration','hifi-duration']]
            # compute mean and std
            time_avg = grp.mean()
            time_std = grp.std()
            # store into avg_times[algo][pct]
            for pct_val in time_avg.index:
                avg_times[algo][pct_val] = {
                    'lofi_duration':       time_avg.loc[pct_val, 'lofi-duration'],
                    'hifi_duration':       time_avg.loc[pct_val, 'hifi-duration'],
                    'lofi_duration_std':   time_std.loc[pct_val, 'lofi-duration'],
                    'hifi_duration_std':   time_std.loc[pct_val, 'hifi-duration'],
                }
        else:
            warnings.warn(f"No timing data for {algo}.")
            continue
        
        for pct in training:
            series_list = results[algo][pct]
            if not series_list:
                raise RuntimeError(f"No data for {algo} at {pct:.1f}%")
            # align on episode, inner join
            df_concat = pd.concat(series_list, axis=1, join="inner")
            # optional: name columns by seed
            df_concat.columns = seeds[: df_concat.shape[1]]
            # compute mean + std
            df_agg = pd.DataFrame({
                "episode": df_concat.index,
                "mean_reward": df_concat.mean(axis=1) if df_concat.shape[1] > 1 else df_concat.iloc[:, 0],
                "std_reward": df_concat.std(axis=1) if df_concat.shape[1] > 1 else 0,
            }).reset_index(drop=True)
            
            # apply moving average across mean rewards and std rewards
            if filter == moving_avg:
                df_agg["mean_reward"] = filter(df_agg, col="mean_reward", window=N)["mean_reward"]
                df_agg["std_reward"]  = filter(df_agg, col="std_reward",  window=N)["std_reward"]
            elif filter == kalman:
                df_agg["mean_reward"] = filter(df_agg, col="mean_reward", proc_var=1e-5, mes_var=1)["mean_reward"]
                df_agg["std_reward"]  = filter(df_agg, col="std_reward",  proc_var=1e-5, mes_var=1)["std_reward"]
            else:
                raise ValueError(f"Unknown filter: {filter}. Use 'moving_avg' or 'kalman'.")
            
            # Drop NaN values
            df_agg.dropna(inplace=True)
            
            aggregated[algo][pct] = df_agg
            
    # persist
    if save_metrics:
        os.makedirs("data", exist_ok=True)
        with open("data/aggregated_results.pkl", "wb") as f:
            pickle.dump(aggregated, f)
        tqdm.write("Aggregated results saved to data/aggregated_results.pkl")
    else:
        tqdm.write("WARNING: Metrics not saved.")

    # plotting
    # compute seed-averaged dropoff & performance metrics
    dropoff_metrics     = metrics.dropoff(aggregated, N=50)
    performance_metrics = metrics.performance(aggregated, avg_times, N=50)
    contour_metrics     = metrics.performance_contour(performance_metrics)
    transfer_metrics    = metrics.transfer_gap(performance_metrics)


    plot.dropoff(dropoff_metrics, save_plot)
    plot.performance(performance_metrics, save_plot)
    plot.performance_contour(contour_metrics, save_plot)
    plot.transfer_gap_plot(transfer_metrics, save_plot)
    plot.training(aggregated, training, save_plot)


if __name__ == "__main__":
    main(N=100, filter=kalman)