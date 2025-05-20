import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from typing import Callable

import glob
from collections import defaultdict

# Local imports
from common import metrics, plot
from common.filters import moving_avg, kalman


# ---- Helper functions ----

def convert_legacy_logs(root_dir: str) -> None:
    """
    Convert legacy `results.csv` files (reward logged every timestep) to the
    new, episode‑level format. The function rewrites the CSV in‑place if it
    detects legacy structure.
    """
    for csv_path in glob.glob(os.path.join(root_dir, "**/logs/results.csv"), recursive=True):
        df = pd.read_csv(csv_path)
        # Legacy ⇒ either a 'timesteps' column or duplicate episode indices.
        if ("timesteps" in df.columns) or df.duplicated(subset=["episode"]).any():
            df = df.groupby("episode", as_index=False)["reward"].sum()
            df.to_csv(csv_path, index=False)


def aggregate_results(seed_results: list[dict]) -> dict:
    """
    Merge per‑seed `results` dicts into a single dict whose `Pbar` dataframes
    contain the mean reward per episode (column `reward`) plus a `reward_std`
    column. Numerical metrics are averaged as scalars.
    """
    merged: dict = {}
    for res in seed_results:
        for algo, lvl_dict in res.items():
            merged.setdefault(algo, {})
            for lvl, data in lvl_dict.items():
                merged[algo].setdefault(lvl, {"dfs": [], "drop": [], "perf": []})
                merged[algo][lvl]["dfs"].append(
                    data["Pbar"][["episode", "reward"]].copy()
                )
                merged[algo][lvl]["drop"].append(data["dropoff"])
                merged[algo][lvl]["perf"].append(data["performance"])

    for algo, lvl_dict in merged.items():
        for lvl, acc in lvl_dict.items():
            rewards = [
                df.set_index("episode")["reward"] for df in acc.pop("dfs")
            ]
            rewards_df = pd.concat(rewards, axis=1)
            mean_df = rewards_df.mean(axis=1).reset_index()
            mean_df.columns = ["episode", "reward"]
            mean_df["reward_std"] = rewards_df.std(axis=1).values
            merged[algo][lvl]["Pbar"] = mean_df
            merged[algo][lvl]["dropoff"] = float(np.mean(acc["drop"]))
            merged[algo][lvl]["performance"] = float(np.mean(acc["perf"]))
    return merged


def aggregate_ctrl(ctrl_list: list[dict]) -> dict:
    """
    Average the baseline statistics (stored in `ctrl`) across seeds so that
    plot helpers keep the same interface: value[0]=mean, value[1]=std.
    """
    tmp = defaultdict(lambda: defaultdict(list))
    for c in ctrl_list:
        for algo, metr in c.items():
            for k, v in metr.items():
                tmp[algo][k].append(v[0] if isinstance(v, list) else v)

    out = {}
    for algo, metr in tmp.items():
        out[algo] = {}
        for k, arr in metr.items():
            out[algo][k] = [float(np.mean(arr)), float(np.std(arr))]
    return out


# ---- Per-seed analysis worker ----

def process_seed(root_dir: str,
                 algos: list[str],
                 training: np.ndarray,
                 N: int) -> tuple[dict, dict]:
    """
    Run the original (single‑seed) analysis pipeline and return its results
    without writing plots/metrics to disk.
    """
    results = {}
    ctrl = {}

    tqdm.write(f"\n### DATA POST‑PROCESSING for {root_dir} ###\n")
    for algo in tqdm(algos, desc="Processing algorithms", position=0):
        dirpath_lofi   = f"{root_dir}LoFi-{algo}/"
        dirpath_hifi   = f"{root_dir}HiFi-{algo}/"
        dirpath_lofi_base = f"{root_dir}baseline/LoFi-{algo}/"
        dirpath_hifi_base = f"{root_dir}baseline/HiFi-{algo}/"
        full_baseline = os.path.exists(dirpath_lofi_base) and os.path.exists(dirpath_hifi_base)

        if not (os.path.exists(dirpath_lofi) and os.path.exists(dirpath_hifi)):
            tqdm.write(f"[WARN] {algo}: missing LoFi/HiFi folders in {root_dir}")
            continue

        results[algo] = {lvl: {} for lvl in training}
        ts_path = f"{root_dir}{algo}_ts.csv"
        if not os.path.exists(ts_path):
            tqdm.write(f"[WARN] {algo}: missing {ts_path}")
            continue
        ts = pd.read_csv(ts_path)

        # ---------- baseline ----------
        baseline_vars = {k: [] for k in ["Pbar_min", "Pbar_max", "T_max", "t_max", "t_end"]}
        base_hifi = pd.read_csv(f"{dirpath_hifi}{algo}_full/logs/results.csv")
        base_hifi = kalman(base_hifi, "reward")
        base_hifi = moving_avg(base_hifi, "reward", window=N)

        max_cond = base_hifi["episode"] >= base_hifi["episode"].iloc[-1] * 0.05
        baseline_vars["Pbar_min"].append(base_hifi["reward"].min())
        baseline_vars["Pbar_max"].append(base_hifi.loc[max_cond, "reward"].max())

        T_hifi = ts.loc[ts["percentage"] == "full", "hifi-duration"].iloc[0]
        ep_max = base_hifi["episode"].iloc[base_hifi.loc[max_cond, "reward"].idxmax()]
        t_end  = base_hifi["episode"].iloc[-1]
        baseline_vars["T_max"].append(ep_max * T_hifi / t_end)
        baseline_vars["t_max"].append(ep_max)
        baseline_vars["t_end"].append(t_end)

        if full_baseline:
            sub_hifi = [
                os.path.join(dirpath_hifi_base, d)
                for d in os.listdir(dirpath_hifi_base)
                if d.startswith("run_")
            ]
            for hifi in sub_hifi:
                df_b = pd.read_csv(f"{hifi}/logs/results.csv")
                df_b = kalman(df_b, "reward")
                df_b = moving_avg(df_b, "reward", window=N)

                max_c = df_b["episode"] >= df_b["episode"].iloc[-1] * 0.05
                baseline_vars["Pbar_min"].append(df_b["reward"].min())
                baseline_vars["Pbar_max"].append(df_b.loc[max_c, "reward"].max())
                baseline_vars["t_max"].append(
                    df_b["episode"].iloc[df_b.loc[max_c, "reward"].idxmax()]
                )
                baseline_vars["t_end"].append(df_b["episode"].iloc[-1])

            ts_base = pd.read_csv(f"{root_dir}baseline/{algo}_ts.csv")
            T_hifi = ts_base["hifi-duration"]
            T_max  = np.multiply(baseline_vars["t_max"], T_hifi / np.array(baseline_vars["t_end"]))
            Pmin, Pmax = np.mean(baseline_vars["Pbar_min"]), np.mean(baseline_vars["Pbar_max"])
            baseline_vars["Pbar_min"] = [Pmin, np.std(baseline_vars["Pbar_min"])]
            baseline_vars["Pbar_max"] = [Pmax, np.std(baseline_vars["Pbar_max"])]
            baseline_vars["T_max"] = [np.mean(T_max), np.std(T_max)]
            baseline_vars["t_max"] = [np.mean(baseline_vars["t_max"]), np.std(baseline_vars["t_max"])]
        else:
            for k in ["Pbar_min", "Pbar_max", "T_max", "t_max"]:
                baseline_vars[k].append(0)

        ctrl[algo] = baseline_vars
        # ---------- training levels ----------
        for lvl in tqdm(training, desc=f"{algo}: training %", position=1, leave=False):
            df_lofi = pd.read_csv(f"{dirpath_lofi}{algo}_{lvl}/logs/results.csv")
            df_hifi = pd.read_csv(f"{dirpath_hifi}{algo}_{lvl}/logs/results.csv")

            df_lofi = df_lofi.groupby("episode").sum().reset_index().drop(columns=["timesteps"], errors="ignore")
            df_hifi = df_hifi.groupby("episode").sum().reset_index().drop(columns=["timesteps"], errors="ignore")

            df_lofi["env"] = 0
            df_hifi["env"] = 1
            df_hifi["episode"] += df_lofi["episode"].max() + 1

            df = pd.concat([df_lofi, df_hifi], ignore_index=True)
            results[algo][lvl]["raw"] = df

            Pbar = moving_avg(kalman(df, "reward"), "reward", window=N)
            results[algo][lvl]["Pbar"] = Pbar

            results[algo][lvl]["time"] = {
                "lofi": ts.loc[ts["percentage"] == str(lvl), "lofi-duration"].iloc[0],
                "hifi": ts.loc[ts["percentage"] == str(lvl), "hifi-duration"].iloc[0],
            }
            results[algo][lvl]["dropoff"] = metrics.dropoff(results[algo][lvl], N)
            results[algo][lvl]["performance"] = metrics.performance(results[algo][lvl], baseline_vars, N)

    return results, ctrl


def main(N: int = 50, save_plot: bool = True, save_metrics: bool = True) -> None:
    """
    Perform full analysis on the results of the training runs.
    """
    algos = ["A2C", "PPO", "SAC", "TD3", "DDPG"]
    training = np.linspace(0, 100, 21)[1:]  # 5…100 %

    base_dir = "experiments/"
    seed_dirs = sorted(d for d in os.listdir(base_dir) if d.startswith("ATC_RL."))
    if not seed_dirs:
        raise RuntimeError("No seed folders found in 'experiments/'. Expected 'ATC_RL.<seed>'.")

    # Ensure every seed uses the new logging format
    for sd in seed_dirs:
        convert_legacy_logs(os.path.join(base_dir, sd))

    # Run the pipeline per seed
    all_results, all_ctrl = [], []
    for sd in seed_dirs:
        seed_root = os.path.join(base_dir, sd) + "/"
        r, c = process_seed(seed_root, algos, training, N)
        all_results.append(r)
        all_ctrl.append(c)

    # Aggregate across seeds
    results = aggregate_results(all_results)
    ctrl    = aggregate_ctrl(all_ctrl)

    tqdm.write("All seeds processed…")

    # Persist aggregated results
    if save_metrics:
        os.makedirs("data", exist_ok=True)
        with open("data/results.pkl", "wb") as f:
            pickle.dump(results, f)
        tqdm.write("Aggregated metrics saved to data/results.pkl")
    else:
        tqdm.write("WARNING: Metrics not saved.")

    # Plot using the aggregated data
    plot.dropoff(results, save_plot)
    plot.performance_contour(results, save_plot)
    plot.transfer_gap_plot(results, save_plot)
    plot.performance(results, save_plot)
    plot.training(results, ctrl, save_plot)


if __name__ == '__main__':
    main()