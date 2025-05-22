import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Set seaborn theme with more appealing aesthetics
sns.set_theme(context='paper', style="whitegrid", palette="husl")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
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
    dropoff_metrics_all_algos: dict, # Added
    metrics_to_plot_config: list[tuple[str, str, str]], # Updated: (base, display, source_key)
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

    all_algo_keys = set()
    if performance_metrics_all_algos:
        all_algo_keys.update(performance_metrics_all_algos.keys())
    if dropoff_metrics_all_algos:
        all_algo_keys.update(dropoff_metrics_all_algos.keys())
    algos = sorted(list(all_algo_keys))

    if not algos:
        warnings.warn("No algorithms found in any provided metrics data. Skipping summary panel.")
        return

    # Determine common training percentages (pct_levels)
    pct_levels = []
    source_for_pct_levels = performance_metrics_all_algos if performance_metrics_all_algos else dropoff_metrics_all_algos
    
    if source_for_pct_levels:
        for algo_name_for_pct in algos: # Iterate through combined list of algos
            algo_data_for_pct = source_for_pct_levels.get(algo_name_for_pct)
            if algo_data_for_pct:
                pct_levels = sorted(list(algo_data_for_pct.keys()))
                break 
    
    if not pct_levels:
        warnings.warn("No training percentage levels found in provided metrics. Skipping summary panel.")
        return

    tqdm.write(f"\n### AGGREGATED PERFORMANCE SUMMARY HEATMAPS (Per Metric) ###\n")

    for metric_base_name, metric_display_name, source_key in tqdm(metrics_to_plot_config, desc="Processing Metrics for Summary Heatmaps"):
        
        fig_height_per_algo = 0.6 
        min_subplot_height = 4    
        plot_height = max(min_subplot_height, len(algos) * fig_height_per_algo + 2.5) # Adjusted for better spacing

        fig, ax = plt.subplots(figsize=(10, plot_height)) 
        
        mean_data_for_heatmap = pd.DataFrame(index=algos, columns=pct_levels, dtype=float)
        annot_data_for_heatmap = pd.DataFrame(index=algos, columns=pct_levels, dtype=object)
        
        has_any_data_for_metric = False
        for algo_name in algos:
            source_dict_for_algo = None
            if source_key == 'performance':
                source_dict_for_algo = performance_metrics_all_algos.get(algo_name, {})
            elif source_key == 'dropoff':
                source_dict_for_algo = dropoff_metrics_all_algos.get(algo_name, {})
            else:
                warnings.warn(f"Metric '{metric_display_name}': Unknown source_key '{source_key}' for metric_base '{metric_base_name}'. Skipping this metric for algo '{algo_name}'.")
                # Fill row with NaN for this algo for this metric if source_key is bad for this metric
                mean_data_for_heatmap.loc[algo_name, :] = np.nan 
                annot_data_for_heatmap.loc[algo_name, :] = "N/A"
                continue 

            for pct_level in pct_levels:
                metrics_for_pct = source_dict_for_algo.get(pct_level, {})
                
                mean_val = metrics_for_pct.get(metric_base_name + '_mean', np.nan)
                std_val = metrics_for_pct.get(metric_base_name + '_std', np.nan)
                
                if pd.notna(mean_val): has_any_data_for_metric = True

                mean_data_for_heatmap.loc[algo_name, pct_level] = mean_val
                
                if pd.notna(mean_val) and pd.notna(std_val):
                    annot_data_for_heatmap.loc[algo_name, pct_level] = f"{mean_val:.1f}±{std_val:.1f}"
                elif pd.notna(mean_val):
                    annot_data_for_heatmap.loc[algo_name, pct_level] = f"{mean_val:.1f}"
                else:
                    annot_data_for_heatmap.loc[algo_name, pct_level] = "N/A"
        
        if not has_any_data_for_metric:
            warnings.warn(f"No data found for metric '{metric_display_name}' across all algorithms and percentages. Skipping its heatmap.")
            plt.close(fig) # Close the empty figure
            continue

        sns.heatmap(mean_data_for_heatmap,
                    annot=annot_data_for_heatmap,
                    fmt='s',
                    cmap='viridis',
                    ax=ax,
                    linewidths=.5,
                    cbar_kws={'label': 'Mean Value'})
        
        ax.set_title(f"Performance Summary: {metric_display_name}", fontweight='bold', fontsize=16)
        ax.set_ylabel('Algorithm', fontsize=13)
        ax.set_xlabel('LoFi Pre-training [%]', fontsize=13)
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.tight_layout()

        if save_plot:
            safe_metric_name = metric_base_name.replace('/', '_').replace('\\', '_') # Sanitize
            plot_filename = f"{plot_filename_prefix}_{safe_metric_name}.png"
            save_path = os.path.join(root_dir, plot_filename)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # tqdm.write(f"Saved performance summary heatmap to {save_path}") # tqdm now handles the outer loop
            plt.close(fig)
        else:
            plt.show()

""" WARNING: DEPRECATED
def transfer_gap_plot(transfer_metrics: dict, save_plot: bool = True, dpi: int = 600) -> None:
    '''
    Plot transfer gap (DeltaPbar mean +/- std) vs. training level for each algorithm.
    transfer_metrics: dict[algo][pct] -> {'mean': value, 'std': value}.
    '''
    root_dir = 'plots/transfer_gap/'
    if not transfer_metrics: return
    algos = list(transfer_metrics.keys())
    training = np.array([])
    for algo_check in algos:
        if transfer_metrics[algo_check]:
            training = np.array(sorted(transfer_metrics[algo_check].keys()))
            break
    if training.size == 0:
        tqdm.write("No training levels found in transfer_metrics to plot.")
        return

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    tqdm.write("\n### TRANSFER GAP PLOTS (Mean +/- Std) ###\n")
    for algo in tqdm(algos, desc="Transfer gap"):
        if not transfer_metrics.get(algo):
            continue
        
        mean_values = np.array([transfer_metrics[algo].get(lvl, {}).get('mean', np.nan) for lvl in training])
        std_values  = np.array([transfer_metrics[algo].get(lvl, {}).get('std', np.nan) for lvl in training])

        if np.all(np.isnan(mean_values)):
            warnings.warn(f"Skipping plot for {algo} - Transfer Gap as all mean values are NaN.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(training, mean_values, marker='o', label=f"{algo} (Mean)")
        ax.fill_between(training, mean_values - std_values, mean_values + std_values, alpha=0.2, label=f"{algo} (±1 Std Dev)")
        
        ax.set_title(f"{algo} Transfer Gap (ΔP̄)", fontweight='bold')
        ax.set_xlabel('Training Level [%]')
        ax.set_ylabel('ΔP̄ (Performance vs Baseline) [-]')
        ax.legend()
        sns.despine()
        ax.grid(True, linestyle='--', alpha=0.7)
        if save_plot:
            plt.savefig(f"{root_dir}{algo}_gap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
"""
def time_to_threshold_stacked_area(threshold_data: dict,
                                   save_plot: bool = True,
                                   dpi: int = 600) -> None:
    """
    Generates stacked area plots showing the relative time to reach different reward thresholds.

    Parameters:
    -----------
    threshold_data: dict
        Output from metrics.calculate_time_to_threshold.
        Structure: threshold_data[algo] -> pd.DataFrame
        DataFrame has pre-training percentages as index, threshold labels as columns,
        and values represent the mean height of the stack layer.
    save_plot: bool
    dpi: int
    """
    root_dir = 'plots/time_to_threshold/'
    if not threshold_data:
        tqdm.write("No data provided for time_to_threshold_stacked_area plot.")
        return
    
    algos = list(threshold_data.keys())
    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    tqdm.write("\n### TIME TO THRESHOLD STACKED AREA PLOTS ###\n")
    for algo in tqdm(algos, desc="Plotting Time to Threshold"):
        df_algo = threshold_data.get(algo)
        if df_algo is None or df_algo.empty:
            warnings.warn(f"No threshold data for {algo}. Skipping plot.")
            continue
        
        df_algo = df_algo.sort_index() # Ensure pre-training levels are sorted
        pct_levels = df_algo.index.values
        
        # Labels for the legend are the column names (threshold_labels)
        # These should be ordered from easiest threshold (bottom of stack) to hardest (top)
        # The calculate_time_to_threshold function sorts thresholds like [-13, -11, ..., -7]
        # So df_algo.columns should already be in this order.
        layer_labels = df_algo.columns.tolist()
        
        valid_layers_data = []
        valid_labels = []
        # The columns in df_algo are already the layer heights.
        for label in layer_labels:
            layer_series = df_algo[label]
            if not layer_series.isnull().all(): # Only include if not all NaN
                valid_layers_data.append(layer_series.fillna(0).values) # Fill NaN with 0 for stackplot layer height
                valid_labels.append(label)
            else:
                warnings.warn(f"Layer '{label}' for algo {algo} is all NaN and will be skipped in the plot.")

        if not valid_layers_data:
            warnings.warn(f"All layers for {algo} are effectively NaN. Skipping plot.")
            continue
            
        # Define colors - example uses viridis.
        # stackplot plots first element in valid_layers_data at bottom.
        # If valid_labels are ["R>=-13", "R>=-11", ...], and viridis is [c1(yellow), c2(green), ...]
        # this will match the example if viridis palette is used directly.
        colors = sns.color_palette("viridis", len(valid_labels))

        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.stackplot(pct_levels, *valid_layers_data, labels=valid_labels, colors=colors, alpha=0.85)
        
        ax.set_title(f"{algo} - Time to Reach Reward Thresholds", fontweight='bold')
        ax.set_xlabel("Pre-training Level [%]")
        ax.set_ylabel("Relative Time to Threshold [-]")
        ax.legend(loc='upper right', title="Reward Threshold") # Match example
        
        # Determine y-limit dynamically based on the sum of layer heights
        # df_algo contains layer heights. Sum them along columns for each pct.
        max_cumulative_y = df_algo.sum(axis=1).max() if not df_algo.empty else 0
        plot_y_max = 1.0 # Default y_max
        if pd.notna(max_cumulative_y) and max_cumulative_y > 0:
            plot_y_max = max(plot_y_max, max_cumulative_y * 1.05) # Add 5% padding
        
        ax.set_ylim(0, plot_y_max)
        if pct_levels.size > 0:
            ax.set_xlim(pct_levels.min(), pct_levels.max())
        else:
            ax.set_xlim(0,100)

        sns.despine()
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{root_dir}{algo}_time_to_threshold.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def training(results: dict,
             training_levels: list,
             save_plot: bool = True,
             dpi: int = 600,
             is_individual_seed: bool = False,
             plot_title_prefix: str = "",
             custom_save_dir: str = None) -> None:
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
    if not results: return

    # tqdm.write("\n### TRAINING PLOTS ###\n") # Moved to analysis.py for individual plots
    for algo in tqdm(results.keys(), desc="Plotting training curves", leave=False): 
        if not results[algo]: continue

        fig, ax = plt.subplots(figsize=(14, 9))
        palette = sns.color_palette("husl", len(training_levels))

        for lvl, color in zip(training_levels, palette):
            if lvl not in results[algo] or results[algo][lvl].empty:
                continue # Skip if no data for this level

            df = results[algo][lvl]
            x = df['episode']
            y = df['mean_reward']
            
            sns.lineplot(x=x, y=y,
                         ax=ax,
                         label=f"{lvl:.0f}% Pre-train", # Clarified label
                         color=color,
                         linewidth=2.0) # Slightly thinner default line
            
            if not is_individual_seed and 'std_reward' in df.columns:
                yerr = df['std_reward']
                ax.fill_between(x,
                                y - yerr,
                                y + yerr,
                                color=color,
                                alpha=0.25) # Slightly more opaque

        title_text = f"{plot_title_prefix}{algo} — Reward vs Episode"
        ax.set_title(title_text, fontweight='bold', fontsize=16)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Reward" if is_individual_seed else "Mean Reward +/- 1 sigma", fontsize=14)
        ax.legend(title="Pre-training Level", loc='best', frameon=True) # loc='best'
        sns.despine()
        ax.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_plot:
            save_filepath = ""
            if custom_save_dir: # Used for individual seed plots
                os.makedirs(custom_save_dir, exist_ok=True)
                # algo name is already part of custom_save_dir structure from analysis.py
                # e.g. custom_save_dir = "plots/training_individual/A2C/seed_123"
                # The file saved is {algo}.png, so "plots/training_individual/A2C/seed_123/A2C.png"
                save_filepath = os.path.join(custom_save_dir, f"{algo}_reward_history.png")
            else: # Original behavior (aggregated plots, if this function is ever called for that)
                os.makedirs(aggregated_root_dir, exist_ok=True)
                save_filepath = os.path.join(aggregated_root_dir, f"{algo}.png")
            
            plt.savefig(save_filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
