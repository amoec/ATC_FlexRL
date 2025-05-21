import os
import numpy as np
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
    Plot dropoff metrics for each algorithm across training levels.

    Parameters:
    -----------
    dropoff_metrics: dict
        Nested dict[algo][pct] -> metrics (floats).
    save_plot: bool
        Whether to save the plot (if False, shows it).
    dpi: int
        Resolution for saved figure.
    '''
    root_dir = 'plots/dropoff/'
    algos = list(dropoff_metrics.keys())
    training = np.array(sorted(dropoff_metrics[algos[0]].keys()))
    plot_vars = {
        'tau':           ('Time-to-Recovery',               'Training Level [%]', r'$\tau$ [ep]'),
        'tau_rel':       ('Relative Time-to-Recovery',      'Training Level [%]', r'$\tau_{rel}$ [-]'),
        'DeltaPbar_d':   ('Dropoff Delta',                  'Training Level [%]', r'$\Delta \bar{P}_d$ [-]'),
        'DeltaPbar_d_rel':('Dropoff Delta Rel. to Switch',  'Training Level [%]', r'$\Delta \bar{P}_d$ [-]'),
        'DeltaPbar_star_d_rel':('Dropoff Delta Rel. to Max','Training Level [%]', r'$\Delta \bar{P}^*_d$ [-]')
    }
    if save_plot:
        for var in plot_vars:
            os.makedirs(f"{root_dir}{var}", exist_ok=True)

    tqdm.write("\n### DROPOFF PLOTS ###\n")
    for algo in tqdm(algos, desc="Algorithms"):
        for var, (title, xlabel, ylabel) in plot_vars.items():
            values = np.array([dropoff_metrics[algo][lvl].get(var, np.nan) for lvl in training])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training, values, marker='o', label=algo)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            sns.despine(left=False, bottom=False)
            ax.grid(True, linestyle='--', alpha=0.7)
            if save_plot:
                plt.savefig(f"{root_dir}{var}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


def performance(perf_metrics: dict, save_plot: bool=True, dpi: int=600) -> None:
    '''
    Plot performance metrics for each algorithm across training levels.

    Parameters:
    -----------
    perf_metrics: dict
        Nested dict[algo][pct] -> performance metrics.
    save_plot: bool
        Whether to save the plot (if False, shows it).
    dpi: int
        Resolution for saved figure.
    '''
    root_dir = 'plots/performance/'
    algos = list(perf_metrics.keys())
    training = np.array(sorted(perf_metrics[algos[0]].keys()))
    plot_vars = {
        'DeltaPbar':    ('Performance Delta',         'Training Level [%]', r'$\Delta\bar{P}$ [-]'),
        'DeltaPbar_rel':('Relative Performance Delta','Training Level [%]', r'$\Delta\bar{P}_{rel}$ [-]'),
        't_XO':         ('Time-to-Crossover',        'Training Level [%]', r'$t_{XO}$ [ep]'),
        't_XO_rel':     ('Rel. Time-to-Crossover',   'Training Level [%]', r'$t_{XO,rel}$ [-]'),
        'T_tot':        ('Total Training Time',      'Training Level [%]', r'$T_{tot}$ [s]'),
        'T_tot_rel':    ('Rel. Total Training Time', 'Training Level [%]', r'$T_{tot,rel}$ [-]')
    }
    if save_plot:
        for var in plot_vars:
            os.makedirs(f"{root_dir}{var}", exist_ok=True)

    tqdm.write("\n### PERFORMANCE PLOTS ###\n")
    for algo in tqdm(algos, desc="Algorithms"):
        for var, (title, xlabel, ylabel) in plot_vars.items():
            values = np.array([perf_metrics[algo][lvl].get(var, np.nan) for lvl in training])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training, values, marker='o', label=algo)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            sns.despine(left=False, bottom=False)
            ax.grid(True, linestyle='--', alpha=0.7)
            if save_plot:
                plt.savefig(f"{root_dir}{var}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


def performance_contour(contour_metrics: dict, save_plot: bool = True, dpi: int = 600) -> None:
    '''
    Plot heatmaps of performance metrics across training levels for each algorithm.

    Parameters:
    -----------
    contour_metrics: dict
        dict[algo] -> DataFrame indexed by pct, columns perf metrics.
    save_plot: bool
        Whether to save the plot (if False, shows it).
    dpi: int
        Resolution for saved figure.
    '''
    root_dir = 'plots/performance_contour/'
    algos = list(contour_metrics.keys())
    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    for algo in tqdm(algos, desc="Performance contour"):
        df = contour_metrics[algo].sort_index()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis', ax=ax)
        ax.set_title(f"{algo} Performance Metrics", fontweight='bold')
        ax.set_xlabel('Training Level [%]')
        plt.tight_layout()
        if save_plot:
            plt.savefig(f"{root_dir}{algo}_contour.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def transfer_gap_plot(transfer_metrics: dict, save_plot: bool = True, dpi: int = 600) -> None:
    '''
    Plot transfer gap (DeltaPbar) vs. training level for each algorithm.

    Parameters:
    -----------
    transfer_metrics: dict
        dict[algo][pct] -> DeltaPbar value.
    save_plot: bool
        Whether to save the plot (if False, shows it).
    dpi: int
        Resolution for saved figure.
    '''
    root_dir = 'plots/transfer_gap/'
    algos = list(transfer_metrics.keys())
    training = np.array(sorted(transfer_metrics[algos[0]].keys()))
    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    for algo in tqdm(algos, desc="Transfer gap"):
        values = np.array([transfer_metrics[algo][lvl] for lvl in training])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(training, values, marker='o')
        ax.set_title(f"{algo} Transfer Gap", fontweight='bold')
        ax.set_xlabel('Training Level [%]')
        ax.set_ylabel('ΔP̄ [-]')
        sns.despine()
        ax.grid(True, linestyle='--', alpha=0.7)
        if save_plot:
            plt.savefig(f"{root_dir}{algo}_gap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def training(results: dict,
             training_levels: list,
             save_plot: bool = True,
             dpi: int = 600) -> None:
    """
    Plot seed-averaged training curves with ±1σ shading.
    
    Parameters:
    -----------
    results: dict
        Nested dict of DataFrames: results[algo][level] → df with
        columns ['episode','mean_reward','std_reward'].
    training_levels: list
        List of levels (e.g. [0,20,…,100]) in the same order as in results.
    save_plot: bool
        Whether to save to disk under plots/training/.
    dpi: int
        Resolution of saved figure.
    """

    root_dir = 'plots/training/'
    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    tqdm.write("\n### TRAINING PLOTS ###\n")
    for algo in tqdm(results.keys(), desc="Plotting training"): 
        fig, ax = plt.subplots(figsize=(14, 9))
        # choose a consistent palette
        palette = sns.color_palette("husl", len(training_levels))

        for lvl, color in zip(training_levels, palette):
            df = results[algo][lvl]
            x = df['episode']
            y = df['mean_reward']
            yerr = df['std_reward']

            sns.lineplot(x=x, y=y,
                         ax=ax,
                         label=f"{lvl:.0f}%",
                         color=color,
                         linewidth=2.5)
            
            # fill +/-1 sigma
            ax.fill_between(x,
                            y - yerr,
                            y + yerr,
                            color=color,
                            alpha=0.3)

        ax.set_title(f"{algo} — Reward vs Episode", fontweight='bold', fontsize=16)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Mean Reward +/- 1 sigma", fontsize=14)
        ax.legend(title="Pre-training Level", loc='upper right', frameon=True)
        sns.despine()
        ax.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{root_dir}{algo}.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
