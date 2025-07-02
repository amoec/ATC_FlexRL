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
         dpi: int = 600,
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
    # store list of DataFrames (episode, reward) per (algo, pct) for metrics
    per_seed_dfs_for_metrics = {algo: defaultdict(list) for algo in algos}
    times = {algo: [] for algo in algos}

    for exp_idx, exp in enumerate(seed_dirs):
        current_seed = seeds[exp_idx]
        exp_dir = os.path.join(base_dir, exp)
        print(f"Processing experiment directory: {exp_dir} (Seed: {current_seed})")

        for algo in algos:
            # extract the timestamps for the current algo
            time_path = os.path.join(exp_dir, f"{algo}_ts.csv")
            
            # load the timestamps
            if os.path.exists(time_path):
                time_df = pd.read_csv(time_path)
                times[algo].append(time_df)
            else:
                warnings.warn(f"Missing timestamps for {algo} in {exp_dir}.")
                # Continue to process reward data even if time data is missing for this seed/algo
            
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
                # print(f"Loading results for {algo} at {pct:.1f}% from {lofi_path} and {hifi_path}") # Too verbose

                if not os.path.exists(lofi_path) or not os.path.exists(hifi_path):
                    # This warning is now per seed, which is fine.
                    warnings.warn(
                        f"Missing results files for seed {current_seed}, {algo} at {pct:.1f}% in {exp_dir}."
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
                    # This check is important. If it fails, better to stop.
                    raise RuntimeError(f"Column 'reward' not found in results.csv for seed {current_seed}, {algo}, {pct:.1f}%")

                # index by episode, store the reward series
                rewards_series = exp_df.set_index("episode")["reward"]
                results[algo][pct].append(rewards_series)

                # Prepare DataFrame for metrics functions (episode, mean_reward)
                # For a single seed, 'mean_reward' is just its 'reward'.
                # metrics.py expects 'mean_reward' column.
                metric_df_single_seed = exp_df[['episode', 'reward']].rename(columns={'reward': 'mean_reward'})
                per_seed_dfs_for_metrics[algo][pct].append(metric_df_single_seed)

    # aggregate across seeds for timing data
    avg_times = {algo: {} for algo in algos}
    for algo in algos:
        if times[algo]:
            # concatenate all seed DataFrames
            time_df_all = pd.concat(times[algo], ignore_index=True)
            # group by percentage
            grp = time_df_all.groupby('percentage')[['lofi-duration','hifi-duration']]
            # compute mean and std
            time_avg = grp.mean()
            time_std = grp.std(ddof=0) # Use ddof=0 if population std is desired, or 1 for sample
            # store into avg_times[algo][pct]
            for pct_val in time_avg.index:
                avg_times[algo][pct_val] = {
                    'lofi_duration_mean':    time_avg.loc[pct_val, 'lofi-duration'],
                    'hifi_duration_mean':    time_avg.loc[pct_val, 'hifi-duration'],
                    'lofi_duration_std':     time_std.loc[pct_val, 'lofi-duration'],
                    'hifi_duration_std':     time_std.loc[pct_val, 'hifi-duration'],
                }
        else:
            warnings.warn(f"No timing data for {algo} to aggregate.")
            # No need to continue for this algo if no time data, as performance metrics depend on it.
            # However, dropoff metrics might still be calculable.
            # For simplicity, we'll let metrics functions handle missing data if avg_times[algo] is empty.

    # Persist per_seed_dfs_for_metrics (raw-ish data for metrics) if needed for debugging
    # if save_metrics:
    #     os.makedirs("data", exist_ok=True)
    #     with open("data/per_seed_dfs_for_metrics.pkl", "wb") as f:
    #         pickle.dump(per_seed_dfs_for_metrics, f)
    #     tqdm.write("Per-seed DFs for metrics saved to data/per_seed_dfs_for_metrics.pkl")

    # Compute metrics (aggregation now happens inside these functions)
    # N_metric = 50 # Window for some internal metric calculations, distinct from filtering N
    dropoff_metrics     = metrics.dropoff(per_seed_dfs_for_metrics, N=N) # Use main N for consistency
    performance_metrics = metrics.performance(per_seed_dfs_for_metrics, avg_times, N=N) # Use main N
    
    # Persist aggregated metrics
    if save_metrics:
        os.makedirs("data", exist_ok=True)
        metrics_to_save = {
            'dropoff': dropoff_metrics,
            'performance': performance_metrics,
            'avg_times': avg_times # Save aggregated times as well
        }
        with open("data/aggregated_metrics.pkl", "wb") as f:
            pickle.dump(metrics_to_save, f)
        tqdm.write("Aggregated metrics saved to data/aggregated_metrics.pkl")
    else:
        tqdm.write("WARNING: Metrics not saved.")


    # Plotting
    # Individual seed training plots
    tqdm.write("\n### INDIVIDUAL SEED TRAINING PLOTS ###\n")
    individual_plot_base_dir = "plots/training_individual"
    os.makedirs(individual_plot_base_dir, exist_ok=True)

    for algo in tqdm(algos, desc="Plotting individual seed training"):
        if not results[algo]:
            warnings.warn(f"No results data to plot for algo {algo}")
            continue
        
        algo_plot_dir = os.path.join(individual_plot_base_dir, algo)
        os.makedirs(algo_plot_dir, exist_ok=True)

        num_seeds_for_algo = 0
        if training.size > 0 and results[algo][training[0]]:
             num_seeds_for_algo = len(results[algo][training[0]])
                
        if num_seeds_for_algo == 0 and any(results[algo][pct] for pct in training): # Check if any pct has data
            # Infer num_seeds from the first pct that has data
            for pct_check in training:
                if results[algo][pct_check]:
                    num_seeds_for_algo = len(results[algo][pct_check])
                    break
        
        if num_seeds_for_algo == 0:
            warnings.warn(f"Could not determine number of seeds for {algo}, skipping individual plots.")
            continue

        for seed_idx in range(num_seeds_for_algo):
            current_seed_id = seeds[seed_idx] # Assumes `seeds` list is correctly ordered and complete
            
            seed_specific_plot_data = {algo: {}}
            has_data_for_this_seed_plot = False
            
            for pct in training:
                if results[algo][pct] and len(results[algo][pct]) > seed_idx:
                    seed_series = results[algo][pct][seed_idx]
                    if seed_series.empty:
                        continue

                    plot_df = pd.DataFrame({
                        'episode': seed_series.index,
                        'mean_reward': seed_series.values,
                        'std_reward': np.zeros_like(seed_series.values) # std is 0 for single seed run
                    })
                    
                    # Apply filter to individual seed's reward curve for plotting
                    if filter == moving_avg:
                        plot_df["mean_reward"] = moving_avg(plot_df.copy(), col="mean_reward", window=N)["mean_reward"]
                    elif filter == kalman:
                        plot_df["mean_reward"] = kalman(plot_df.copy(), col="mean_reward", proc_var=1e-5, mes_var=1)["mean_reward"]
                    else:
                        # No filter or unknown filter, plot raw
                        pass # plot_df["mean_reward"] is already the raw reward
                    
                    plot_df.dropna(inplace=True) # After filtering

                    if not plot_df.empty:
                        seed_specific_plot_data[algo][pct] = plot_df
                        has_data_for_this_seed_plot = True
            
            if has_data_for_this_seed_plot:
                seed_plot_dir = os.path.join(algo_plot_dir, f"seed_{current_seed_id}")
                poster_dir = 'plots/poster/training/'
                # The plot.training function will create {algo}.png inside this directory.
                # So, the file will be like plots/training_individual/A2C/seed_123/A2C.png
                plot.training(
                    seed_specific_plot_data, # Contains data for one algo, one seed, all pcts
                    training,
                    save_plot=save_plot,
                    dpi=600,
                    is_individual_seed=True,
                    plot_title_prefix=f"Seed {current_seed_id} - ", # Algo name will be added by plot.training
                    custom_save_dir=seed_plot_dir,
                    poster_format=False
                )
                plot.poster(
                    seed_specific_plot_data, # Contains data for one algo, one seed, all pcts
                    training,
                    save_plot=save_plot,
                    dpi=600,
                    custom_save_dir=poster_dir,
                    xlim=(0, max_ep), # Set xlim to max_ep for poster plots
                )
    
    # Plots for aggregated metrics
    # contour_metrics     = metrics.performance_contour(performance_metrics) # This line is no longer needed for the new heatmap plots
    # transfer_metrics    = metrics.transfer_gap(performance_metrics) 
    num_time_thresholds = 10 
    time_to_threshold_metrics = metrics.calculate_time_to_threshold(
        per_seed_dfs_for_metrics, 
        num_thresholds=num_time_thresholds,
        max_ep_for_run=max_ep
    )

    # Define which metrics to include in the heatmaps and their display names
    # This config can be used for both individual and summary panel plots, or you can have separate ones.
    metrics_for_heatmaps_config = [
        ('DeltaPbar',       'Absolute Change in Final Reward',     'performance'),
        ('T_tot_rel',       'Relative Total Training Time [%]', 'performance'),
        ('DeltaPbar_star',  'Absolute Transfer Penalty',       'dropoff')
    ]

    plot.dropoff(dropoff_metrics, save_plot)
    plot.performance(performance_metrics, save_plot)
    
    # Call the new individual heatmaps plot function
    plot.plot_individual_performance_heatmaps(
        performance_metrics_all_algos=performance_metrics,
        dropoff_metrics_all_algos=dropoff_metrics, # Pass dropoff_metrics
        metrics_to_plot_config=metrics_for_heatmaps_config,
        save_plot=save_plot,
        dpi=dpi
    )

    # Call the new summary panel plot function
    plot.plot_performance_summary_panel(
        performance_metrics_all_algos=performance_metrics,
        dropoff_metrics_all_algos=dropoff_metrics, # Pass dropoff_metrics
        metrics_to_plot_config=metrics_for_heatmaps_config, 
        save_plot=save_plot,
        dpi=dpi,
        plot_filename_prefix="all_algos_summary" # Prefix for the filenames
    )
    
if __name__ == "__main__":
    main(N=50, filter=kalman, dpi=600)