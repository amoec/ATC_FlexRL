import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set seaborn theme with more appealing aesthetics
sns.set_theme(context='paper', style="whitegrid", palette="husl")
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.size': 20,
    # 'axes.titlesize': 30,
    # 'axes.labelsize': 20,
    # 'legend.fontsize': 18,
    # 'xtick.labelsize': 18,
    # 'ytick.labelsize': 18
})

def dropoff(dropoff_metrics: dict, save_plot: bool=True, dpi: int=600) -> None:
    '''
    Plot dropoff metrics (mean +/- std) for each algorithm across training levels.
    '''
    root_dir = 'plots/dropoff/'
    if not dropoff_metrics: return
    algos = list(dropoff_metrics.keys())
    # Infer training levels from the first algo that has data
    training = np.array([])
    for algo_check in algos:
        if dropoff_metrics[algo_check]:
            training = np.array(sorted(dropoff_metrics[algo_check].keys()))
            break
    if training.size == 0:
        tqdm.write("No training levels found in dropoff_metrics to plot.")
        return

    plot_vars = { # Base names of metrics
        'tau':           ('Time-to-Recovery',               'Training Level [%]', r'$\tau$ [ep]'),
        'tau_rel':       ('Relative Time-to-Recovery',      'Training Level [%]', r'$\tau_{rel}$ [-]'),
        'DeltaPbar_d':   ('Dropoff Delta',                  'Training Level [%]', r'$\Delta \bar{P}_d$ [-]'),
        'DeltaPbar_d_rel':('Dropoff Delta Rel. to Switch',  'Training Level [%]', r'$\Delta \bar{P}_d$ [%]'), # Consider renaming ylabel if ambiguous
        'DeltaPbar_star_d_rel':('Transfer Penalty','Episodes in LoFi Environment [%]', r'$\Delta \bar{P}^*_d$ [%]') # Same here
    }
    if save_plot:
        for var_base_name in plot_vars:
            os.makedirs(f"{root_dir}{var_base_name}", exist_ok=True)

    tqdm.write("\n### DROPOFF PLOTS (Mean +/- Std) ###\n")
    for algo in tqdm(algos, desc="Algorithms (Dropoff)"):
        if not dropoff_metrics.get(algo):
            continue
        for var_base_name, (title, xlabel, ylabel) in plot_vars.items():
            mean_values = np.array([dropoff_metrics[algo].get(lvl, {}).get(var_base_name + '_mean', np.nan) for lvl in training])
            std_values  = np.array([dropoff_metrics[algo].get(lvl, {}).get(var_base_name + '_std', np.nan) for lvl in training])
            
            # Skip plotting if all means are NaN
            if np.all(np.isnan(mean_values)):
                warnings.warn(f"Skipping plot for {algo} - {var_base_name} as all mean values are NaN.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training, mean_values, marker='o', label=f"{algo} (Mean)")
            ax.fill_between(training, mean_values - std_values, mean_values + std_values, alpha=0.2, label=f"{algo} (±1 Std Dev)")
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            sns.despine(left=False, bottom=False)
            ax.grid(True, linestyle='--', alpha=0.7)
            if save_plot:
                plt.savefig(f"{root_dir}{var_base_name}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


def performance(perf_metrics: dict, save_plot: bool=True, dpi: int=600) -> None:
    '''
    Plot performance metrics (mean +/- std) for each algorithm across training levels.
    '''
    root_dir = 'plots/performance/'
    if not perf_metrics: return
    algos = list(perf_metrics.keys())
    training = np.array([])
    for algo_check in algos:
        if perf_metrics[algo_check]:
            training = np.array(sorted(perf_metrics[algo_check].keys()))
            break
    if training.size == 0:
        tqdm.write("No training levels found in perf_metrics to plot.")
        return
        
    plot_vars = { # Base names
        'DeltaPbar':    ('Performance Delta',         'Training Level [%]', r'$\Delta\bar{P}$ [-]'),
        'DeltaPbar_rel':('Relative Performance Delta','Training Level [%]', r'$\Delta\bar{P}_{rel}$ [-]'),
        't_XO':         ('Time-to-Crossover',        'Training Level [%]', r'$t_{XO}$ [ep]'),
        't_XO_rel':     ('Rel. Time-to-Crossover',   'Training Level [%]', r'$t_{XO,rel}$ [-]'),
        'T_tot':        ('Total Training Time until XO',      'Training Level [%]', r'$T_{tot,XO}$ [s]'),
        'T_tot_rel':    ('Rel. Total Training Time until XO', 'Training Level [%]', r'$T_{tot,XO,rel}$ [-]')
    }
    if save_plot:
        for var_base_name in plot_vars:
            os.makedirs(f"{root_dir}{var_base_name}", exist_ok=True)

    tqdm.write("\n### PERFORMANCE PLOTS (Mean +/- Std) ###\n")
    for algo in tqdm(algos, desc="Algorithms (Performance)"):
        if not perf_metrics.get(algo):
            continue
        for var_base_name, (title, xlabel, ylabel) in plot_vars.items():
            mean_values = np.array([perf_metrics[algo].get(lvl, {}).get(var_base_name + '_mean', np.nan) for lvl in training])
            std_values  = np.array([perf_metrics[algo].get(lvl, {}).get(var_base_name + '_std', np.nan) for lvl in training])

            if np.all(np.isnan(mean_values)):
                warnings.warn(f"Skipping plot for {algo} - {var_base_name} as all mean values are NaN.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training, mean_values, marker='o', label=f"{algo} (Mean)")
            ax.fill_between(training, mean_values - std_values, mean_values + std_values, alpha=0.2, label=f"{algo} (±1 Std Dev)")
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            sns.despine(left=False, bottom=False)
            ax.grid(True, linestyle='--', alpha=0.7)
            if save_plot:
                plt.savefig(f"{root_dir}{var_base_name}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


def plot_individual_performance_heatmaps(
    performance_metrics_all_algos: dict,
    dropoff_metrics_all_algos: dict, # Added
    metrics_to_plot_config: list[tuple[str, str, str]], # Updated: (base, display, source_key)
    save_plot: bool = True,
    dpi: int = 600
) -> None:
    """
    Plots individual heatmaps for each algorithm, showing selected performance metrics.
    - Takes full performance_metrics and dropoff_metrics dicts.
    - metrics_to_plot_config: list of (metric_base_name, display_name, source_key).
      source_key is 'performance' or 'dropoff'.
    - If the number of configured metrics is odd and >= 3, the middle metric is removed.
    - Annotates cells with 'mean ± std'.
    """
    root_dir = 'plots/performance_heatmaps_individual/'
    # Combine checks for primary data sources
    if not performance_metrics_all_algos and not dropoff_metrics_all_algos:
        warnings.warn("No performance or dropoff metrics provided for individual heatmaps.")
        return
    if not metrics_to_plot_config:
        warnings.warn("No metrics_to_plot_config specified for individual heatmaps.")
        return

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    tqdm.write("\n### INDIVIDUAL PERFORMANCE HEATMAPS ###\n")

    # Determine common training percentages (pct_levels)
    # Prefer performance_metrics for defining pct_levels, fallback to dropoff_metrics
    source_for_pct_levels = performance_metrics_all_algos if performance_metrics_all_algos else dropoff_metrics_all_algos
    
    pct_levels = []
    if source_for_pct_levels:
        for algo_name_for_pct, algo_data_for_pct in source_for_pct_levels.items():
            if algo_data_for_pct: # Check if the algo's metric dict is not empty
                pct_levels = sorted(list(algo_data_for_pct.keys()))
                break
    
    if not pct_levels:
        warnings.warn("No training percentage levels found in provided metrics. Skipping individual heatmaps.")
        return

    # Iterate over algos present in performance_metrics primarily,
    # or all algos if performance_metrics is empty but dropoff_metrics is not.
    # This ensures plots are generated if at least one data source has the algo.
    algo_keys_to_iterate = set()
    if performance_metrics_all_algos:
        algo_keys_to_iterate.update(performance_metrics_all_algos.keys())
    if dropoff_metrics_all_algos:
         algo_keys_to_iterate.update(dropoff_metrics_all_algos.keys())
    
    if not algo_keys_to_iterate:
        warnings.warn("No algorithms found in any provided metrics data. Skipping individual heatmaps.")
        return

    for algo_name in tqdm(sorted(list(algo_keys_to_iterate)), desc="Individual Algo Heatmaps"):
        # Apply "remove middle column" logic
        current_metrics_config_full = list(metrics_to_plot_config) 
        num_initial_metrics = len(current_metrics_config_full)
        if num_initial_metrics >= 3 and num_initial_metrics % 2 == 1:
            middle_index = num_initial_metrics // 2
            removed_metric_info = current_metrics_config_full.pop(middle_index)
            # warnings.warn(f"For {algo_name}, removed middle metric '{removed_metric_info[1]}' from display.")
        
        if not current_metrics_config_full:
            warnings.warn(f"No metrics left to display for {algo_name} after filtering. Skipping its heatmap.")
            continue

        metric_display_names_for_plot = [mc[1] for mc in current_metrics_config_full]
        
        mean_data_for_heatmap = pd.DataFrame(index=pct_levels, columns=metric_display_names_for_plot, dtype=float)
        annot_data_for_heatmap = pd.DataFrame(index=pct_levels, columns=metric_display_names_for_plot, dtype=object)

        has_any_data_for_algo = False
        for pct_level in pct_levels:
            for metric_base_name, metric_display_name, source_key in current_metrics_config_full:
                source_dict_for_algo = None
                if source_key == 'performance':
                    source_dict_for_algo = performance_metrics_all_algos.get(algo_name, {})
                elif source_key == 'dropoff':
                    source_dict_for_algo = dropoff_metrics_all_algos.get(algo_name, {})
                else:
                    warnings.warn(f"Algo {algo_name}, Pct {pct_level}: Unknown source_key '{source_key}' for metric '{metric_base_name}'. Skipping.")
                    mean_data_for_heatmap.loc[pct_level, metric_display_name] = np.nan
                    annot_data_for_heatmap.loc[pct_level, metric_display_name] = "N/A"
                    continue
                
                metrics_for_pct = source_dict_for_algo.get(pct_level, {}) # Handles if algo or pct is missing in source

                mean_val = metrics_for_pct.get(metric_base_name + '_mean', np.nan)
                std_val = metrics_for_pct.get(metric_base_name + '_std', np.nan)
                
                if pd.notna(mean_val): has_any_data_for_algo = True

                mean_data_for_heatmap.loc[pct_level, metric_display_name] = mean_val
                
                if pd.notna(mean_val) and pd.notna(std_val):
                    annot_data_for_heatmap.loc[pct_level, metric_display_name] = f"{mean_val:.1f}±{std_val:.1f}"
                elif pd.notna(mean_val):
                    annot_data_for_heatmap.loc[pct_level, metric_display_name] = f"{mean_val:.1f}"
                else:
                    annot_data_for_heatmap.loc[pct_level, metric_display_name] = "N/A"
        
        if not has_any_data_for_algo:
            warnings.warn(f"No data found for any configured metric for algorithm {algo_name}. Skipping its heatmap.")
            continue
        if mean_data_for_heatmap.empty: # Should be caught by has_any_data_for_algo
            warnings.warn(f"Resulting heatmap data is empty for {algo_name}. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(metric_display_names_for_plot) * 3.5), 8)) # Increased width per metric
        sns.heatmap(mean_data_for_heatmap.sort_index(), 
                    annot=annot_data_for_heatmap.sort_index(),
                    fmt='s',
                    cmap='viridis',
                    square=True,
                    ax=ax,
                    linewidths=.5,
                    cbar_kws={'label': 'Mean Metric Value'})
        
        ax.set_title(f"{algo_name} - Performance Summary", fontweight='bold')
        ax.set_ylabel('LoFi Pre-training [%]')
        ax.set_xlabel('Performance Metric')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{root_dir}{algo_name}_performance_heatmap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def plot_performance_summary_panel(
    performance_metrics_all_algos: dict,
    dropoff_metrics_all_algos: dict,  # Added
    metrics_to_plot_config: list[tuple[str, str, str]],  # (base, display, source_key)
    save_plot: bool = True,
    dpi: int = 600,
    plot_filename_prefix: str = "all_algos_performance_summary"
) -> None:
    """
    Generates a separate heatmap figure for each specified performance metric.
    Each heatmap shows Algorithms vs. Training Percentages for that metric.
    Cells display 'mean ± std' and are colored by the mean value.
    metrics_to_plot_config: list of (metric_base_name, display_name, source_key).
    """
    root_dir = 'plots/performance_summary_panel/'
    if not performance_metrics_all_algos and not dropoff_metrics_all_algos:
        warnings.warn("No performance or dropoff metrics provided for summary panel.")
        return
    if not metrics_to_plot_config:
        warnings.warn("No metrics_to_plot_config specified for summary panel.")
        return

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    # collect all algos present
    all_algo_keys = set()
    all_algo_keys.update(performance_metrics_all_algos.keys())
    all_algo_keys.update(dropoff_metrics_all_algos.keys())
    if not all_algo_keys:
        warnings.warn("No algorithms found in any provided metrics data. Skipping summary panel.")
        return

    # force order: on-policy first, then off-policy
    on_policy  = ["A2C", "PPO"]
    off_policy = ["DDPG", "SAC", "TD3"]
    on_filtered  = [a for a in on_policy  if a in all_algo_keys]
    off_filtered = [a for a in off_policy if a in all_algo_keys]
    algos = on_filtered + off_filtered
    num_on = len(on_filtered)

    # Determine common training percentages (pct_levels)
    pct_levels = []
    source_for_pct_levels = performance_metrics_all_algos or dropoff_metrics_all_algos
    for algo_name in algos:
        algo_data = source_for_pct_levels.get(algo_name, {})
        if algo_data:
            pct_levels = sorted(algo_data.keys())
            break
    if not pct_levels:
        warnings.warn("No training percentage levels found. Skipping summary panel.")
        return

    tqdm.write("\n### AGGREGATED PERFORMANCE SUMMARY HEATMAPS (Per Metric) ###\n")

    for metric_base_name, metric_display_name, source_key in tqdm(
        metrics_to_plot_config, desc="Processing Metrics for Summary Heatmaps"
    ):
        fig, ax = plt.subplots(figsize=(12, 12))
        mean_df = pd.DataFrame(index=algos, columns=pct_levels, dtype=float)
        annot_df = pd.DataFrame(index=algos, columns=pct_levels, dtype=object)

        has_data = False
        for algo_name in algos:
            # skip any missing row (shouldn't happen here)
            perf_dict = (performance_metrics_all_algos if source_key=='performance' else dropoff_metrics_all_algos)
            source_dict = perf_dict.get(algo_name, {})
            for pct in pct_levels:
                m = source_dict.get(pct, {})
                mean_val = m.get(metric_base_name + '_mean', np.nan)
                std_val  = m.get(metric_base_name + '_std',  np.nan)
                if pd.notna(mean_val):
                    has_data = True
                mean_df.loc[algo_name, pct] = mean_val
                if pd.notna(mean_val) and pd.notna(std_val):
                    annot_df.loc[algo_name, pct] = f"{mean_val:.1f}±{std_val:.1f}"
                elif pd.notna(mean_val):
                    annot_df.loc[algo_name, pct] = f"{mean_val:.1f}"
                else:
                    annot_df.loc[algo_name, pct] = "N/A"

        if not has_data:
            warnings.warn(f"No data for metric '{metric_display_name}'. Skipping.")
            plt.close(fig)
            continue

        sns.heatmap(
            mean_df,
            annot=annot_df,
            fmt='s',
            cmap='viridis',
            square=True,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'label': 'Mean Value'}
        )
        # draw thin white divider between on- and off-policy blocks
        if num_on and len(algos) > num_on:
            ax.hlines(
                y=num_on,
                xmin=-0.5,
                xmax=len(pct_levels),
                colors='white',
                linewidth=5,
                zorder=2
            )

        ax.set_title(metric_display_name, fontweight='bold', fontsize=30)
        ax.set_ylabel('Algorithm', fontsize=26)
        ax.set_xlabel('Episodes in LoFi [%]', fontsize=26, zorder=1)
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        if save_plot:
            safe_name = metric_base_name.replace('/', '_').replace('\\', '_')
            fname = f"{plot_filename_prefix}_{safe_name}.png"
            plt.savefig(os.path.join(root_dir, fname), dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
def training(results: dict,
             training_levels: list,
             save_plot: bool = True,
             dpi: int = 600,
             is_individual_seed: bool = False,
             plot_title_prefix: str = "",
             custom_save_dir: str = None,
             poster_format: bool = False) -> None:
    """
    Plot training curves. 
    If is_individual_seed=True, plots for a single seed.
    Otherwise, plots seed-averaged training curves with ±1σ shading (original behavior).
    
    Parameters:
    -----------
    results: dict
        If not individual: results[algo][level] -> df with ['episode','mean_reward','std_reward'].
        If individual: results[algo_name_of_current_seed_plot][level] -> df with ['episode','mean_reward'] (std_reward is 0 or ignored).
    training_levels: list
        List of levels (e.g. [0,20,…,100]).
    save_plot: bool
    dpi: int
    is_individual_seed: bool
        If True, adapts title, save path, and std dev handling.
    plot_title_prefix: str
        Prefix for the plot title, e.g., "Seed X - ". Algo name added after.
    custom_save_dir: str
        If provided (for individual seeds), plots are saved as {algo}.png within this directory.
    """

    # This root_dir is for the aggregated case, not used if custom_save_dir is set.
    aggregated_root_dir = 'plots/training/'
    if poster_format:
        aggregated_root_dir = 'plots/poster/training/'
        
    if not results: return

    # tqdm.write("\n### TRAINING PLOTS ###\n") # Moved to analysis.py for individual plots
    for algo in tqdm(results.keys(), desc="Plotting training curves", leave=False): 
        if not results[algo]: continue

        fig, ax = plt.subplots(figsize=(9, 9)) if not poster_format else plt.subplots(figsize=(4,4)) # Poster format is square
        palette = sns.color_palette("husl", len(training_levels))

        for lvl, color in zip(training_levels, palette):
            if lvl not in results[algo] or results[algo][lvl].empty:
                continue # Skip if no data for this level

            df = results[algo][lvl]
            x = df['episode']
            y = df['mean_reward']
            
            sns.lineplot(x=x, 
                         y=y,
                         ax=ax,
                         label=f"{lvl:.0f} %" if not poster_format else None, # Clarified label
                         color=color,
                         linewidth=5.0 if not poster_format else 5, # Thicker line for poster format
                         ) # Slightly thinner default line
            
            if not is_individual_seed and 'std_reward' in df.columns and not poster_format:
                yerr = df['std_reward']
                ax.fill_between(x,
                                y - yerr,
                                y + yerr,
                                color=color,
                                alpha=0.25) # Slightly more opaque

        title_text = f"{algo} Reward vs Episode"
        if not poster_format:
            ax.set_title(title_text, fontweight='bold', fontsize=30)
        ax.set_xlabel("Episode", fontsize=26 if not poster_format else 16)
        ax.set_ylabel("Reward" if is_individual_seed or poster_format else "Mean Reward +/- 1 sigma", fontsize=26 if not poster_format else 16)
        if poster_format:
            ax.set_ylim(-35, 0)
            ax.set_xlim(0, 500) # Adjusted for poster format
        else:
            ax.legend(title="LoFi Episodes", loc='best', frameon=True, fontsize=18, title_fontsize=20)
        sns.despine()
        ax.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_plot:
            save_filepath = ""
            if custom_save_dir: # Used for individual seed plots
                if poster_format:
                    os.makedirs(aggregated_root_dir, exist_ok=True)
                    save_filepath = os.path.join(aggregated_root_dir, f"{algo}_reward_history.png")
                else:
                    os.makedirs(custom_save_dir, exist_ok=True)
                    # algo name is already part of custom_save_dir structure from analysis.py
                # e.g. custom_save_dir = "plots/training_individual/A2C/seed_123"
                # The file saved is {algo}.png, so "plots/training_individual/A2C/seed_123/A2C.png"
                    save_filepath = os.path.join(custom_save_dir if not poster_format else aggregated_root_dir, f"{algo}_reward_history.png")
            else: # Original behavior (aggregated plots, if this function is ever called for that)
                os.makedirs(aggregated_root_dir, exist_ok=True)
                save_filepath = os.path.join(aggregated_root_dir, f"{algo}.png")
            
            plt.savefig(save_filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def poster(
    results: dict,
    training_levels: list, # Used for consistent color mapping across levels (though not for red/blue line)
    save_plot: bool = True,
    dpi: int = 600,
    custom_save_dir: str = None, # If provided, this is the root for algo-specific dirs
    xlim: tuple = (0, 1000) # Default x-axis limit for poster plots
) -> None:
    """
    Generate individual poster-style plots for each training level of each algorithm.
    Plots are saved under {output_dir}/{algo_name}/{algo_name}_lvl_{lvl}_poster.png.
    The first 'lvl'% of the x-axis (defined by xlim[1]) is plotted in red, 
    and the rest (from the lvl% point onwards) is plotted in blue.
    
    Parameters:
    -----------
    results: dict
        Results dictionary structured as {algo_name: {level: df}}.
    training_levels: list
        List of all possible training levels (e.g., [0, 20, ..., 100]).
        (Not directly used for the red/blue line color in this version).
    save_plot: bool
        Whether to save the plot.
    dpi: int
        DPI for saving the plot.
    custom_save_dir: str, optional
        If provided, this directory will be used as the root to store algorithm-specific
        subdirectories. Defaults to 'plots/poster/'.
    xlim: tuple
        The x-axis limits (min_episode, max_episode) for the plot. The 'lvl'% split
        is calculated based on xlim[1].
    """
    # Set font size and style for poster
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,  # Larger font for poster
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    output_base_dir = custom_save_dir if custom_save_dir is not None else 'plots/poster/'
    
    if not results: return
    
    # The following palette and level_to_color map are not used for the red/blue line logic
    # but are kept if they might be used for other plot elements in the future.
    palette = sns.color_palette("husl", len(training_levels) if training_levels is not None and len(training_levels) > 0 else 1)
    level_to_color = {
        level_val: palette[i] for i, level_val in enumerate(training_levels)
    } if training_levels is not None and len(training_levels) > 0 else {}

    for algo in tqdm(results.keys(), desc="Plotting poster training curves per level", leave=False):
        if not results[algo]: continue
        
        algo_dir_path = os.path.join(output_base_dir, str(algo))
        if save_plot:
            os.makedirs(algo_dir_path, exist_ok=True)
            
        actual_levels_for_algo = sorted(results[algo].keys())

        for lvl in actual_levels_for_algo: # lvl is the percentage value, e.g., 0, 20, 100
            df_orig = results[algo][lvl]
            if df_orig.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Ensure data is sorted by episode for correct splitting and plotting
            df_s = df_orig.sort_values(by='episode').reset_index(drop=True)
            
            x_coords = df_s['episode'].values
            y_coords = df_s['mean_reward'].values

            if len(x_coords) == 0: # Should be caught by df_orig.empty, but as a safeguard
                plt.close(fig)
                continue

            # Calculate the episode value at which the color split occurs (integer)
            split_episode_threshold = np.floor((lvl / 100.0) * xlim[1])

            if len(x_coords) == 1: # Single data point
                x_pt, y_pt = x_coords[0], y_coords[0]
                point_color = 'blue' # Default for lvl=0 or point in blue region
                if lvl == 100: # All red
                    point_color = 'red'
                elif lvl > 0 and x_pt <= split_episode_threshold: # In red part and red part exists
                    point_color = 'red'
                ax.plot(x_pt, y_pt, color=point_color, marker='o', linestyle='None', markersize=5)
            else: # Multiple data points
                if lvl == 0: # All blue
                    ax.plot(x_coords, y_coords, color='blue', linewidth=5)
                elif lvl == 100: # All red
                    ax.plot(x_coords, y_coords, color='red', linewidth=5)
                else: # 0 < lvl < 100, mixed colors
                    # Data for the red part (episodes <= threshold)
                    mask_red_pts = x_coords <= split_episode_threshold
                    x_red_segment = x_coords[mask_red_pts]
                    y_red_segment = y_coords[mask_red_pts]

                    # Data for the blue part (episodes >= threshold)
                    mask_blue_pts = x_coords >= split_episode_threshold
                    x_blue_segment = x_coords[mask_blue_pts]
                    y_blue_segment = y_coords[mask_blue_pts]

                    # Plot red segment if it exists
                    if len(x_red_segment) > 0:
                        if len(x_red_segment) >= 2:
                            ax.plot(x_red_segment, y_red_segment, color='red', linewidth=5)
                        elif len(x_red_segment) == 1: # Single point for red segment
                            ax.plot(x_red_segment[0], y_red_segment[0], color='red', marker='o', linestyle='None', markersize=5)
                    
                    # Plot blue segment if it exists
                    if len(x_blue_segment) > 0:
                        if len(x_blue_segment) >= 2:
                            ax.plot(x_blue_segment, y_blue_segment, color='white', linewidth=5)
                        elif len(x_blue_segment) == 1: # Single point for blue segment
                            ax.plot(x_blue_segment[0], y_blue_segment[0], color='white', marker='o', linestyle='None', markersize=5)
            
            ax.set_xlim(xlim)
            # ax.set_xlabel("Episode", fontsize=18) # Optional: uncomment if needed
            # ax.set_ylabel("Reward", fontsize=18)  # Optional: uncomment if needed
            # ax.grid(linestyle='-', alpha=0.85)   # Optional: uncomment if needed
            plt.tight_layout()

            if save_plot:
                save_filename = f"{algo}_lvl_{lvl}_poster.png"
                save_filepath = os.path.join(algo_dir_path, save_filename)
                plt.savefig(save_filepath, dpi=dpi, bbox_inches='tight')
            else:
                plt.show()
            plt.close(fig)