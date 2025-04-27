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

def dropoff(results: dict, save_plot: bool=True, dpi: int=600) -> None:
    '''
    Plot the full dropoff analysis.

    Parameters:
    -----------
    results: dict
        Full results dictionary.
    save_plot: bool
        Whether to save the plot (shown if False).
    dpi: int
        Dots per inch for the plot.

    Returns:
    --------
    None
    '''
    # Init root dir
    root_dir = 'plots/dropoff/'
    # Init iterables
    algos = list(results.keys())
    training = np.array(list(results[algos[0]].keys()))
    plot_vars = {
        'tau':{
            'title': 'Time-to-Recovery',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\tau$ [ep]'
            },
        'tau_rel': {
            'title': 'Relative Time-to-Recovery',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\tau_{rel}$ [-]'
            },
        'DeltaPbar_d': {
            'title': 'Dropoff Delta',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\Delta \bar{P}_d$ [-]'
            },
        'DeltaPbar_d_rel': {
            'title': 'Dropoff Delta Relative to Switch',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\Delta \bar{P}_d$ [-]'
            },
        'DeltaPbar*_d_rel': {
            'title': 'Dropoff Delta Relative to Max',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\Delta \bar{P}^*_d$ [-]'
            }
        }

    # Create dirs
    if save_plot:
        for var in plot_vars.keys():
            os.makedirs(f"{root_dir}{var}", exist_ok=True)

    tqdm.write("\n### DROPOFF ANALYSIS ###\n")
    for algo in tqdm(algos, desc="Processing algorithms", position=0):
        # Plot the results
        for var in plot_vars.keys():
            data = np.array([results[algo][lvl]['dropoff'][var] for lvl in training])

            # Create DataFrame for seaborn
            df = pd.DataFrame({
                'Training Level': training,
                plot_vars[var]['ylabel']: data,
                'Algorithm': [algo] * len(training)
            })

            # Init the plot with seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df,
                x='Training Level',
                y=plot_vars[var]['ylabel'],
                hue='Algorithm',
                marker='X',
                s=150,
                ax=ax
            )

            # Add regression line
            sns.regplot(
                data=df,
                x='Training Level',
                y=plot_vars[var]['ylabel'],
                scatter=False,
                ci=None,
                line_kws={'linestyle':'--', 'alpha':0.7},
                ax=ax
            )

            # Styling
            plt.title(plot_vars[var]['title'], fontweight='bold')
            plt.xlabel(plot_vars[var]['xlabel'])
            plt.ylabel(plot_vars[var]['ylabel'])
            sns.despine(left=False, bottom=False)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save or show the plot
            if save_plot:
                plt.savefig(f"{root_dir}{var}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

def performance(results: dict, save_plot: bool=True, dpi: int=600) -> None:
    '''
    Plot the full performance analysis.

    Parameters:
    -----------
    results: dict
        Full results dictionary.
    save_plot: bool
        Whether to save the plot (shown if False).
    dpi: int
        Dots per inch for the plot.

    Returns:
    --------
    None
    '''
    # Init root dir
    root_dir = 'plots/performance/'
    # Init iterables
    algos = list(results.keys())
    training = np.array(list(results[algos[0]].keys()))
    plot_vars = {
        'DeltaPbar': {
            'title': 'Performance Delta',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\Delta\bar{P}$ [-]'
            },
        'DeltaPbar_rel': {
            'title': 'Relative Performance Delta',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$\Delta\bar{P}_{rel}$ [-]'
            },
        't_XO': {
            'title': 'Time-to-Crossover',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$t_{XO}$ [ep]'
            },
        't_XO_rel': {
            'title': 'Relative Time-to-Crossover',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$t_{XO,\,rel}$ [-]'
            },
        'T_tot': {
            'title': 'Total Training Time',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$T_{tot}$ [s]'
            },
        'T_tot_rel': {
            'title': 'Relative Total Training Time',
            'xlabel': 'Training Level [%]',
            'ylabel': r'$T_{tot,\,rel}$ [-]'
            }
        }

    # Create dirs
    if save_plot:
        for var in plot_vars.keys():
            os.makedirs(f"{root_dir}{var}", exist_ok=True)

    tqdm.write("\n### PERFORMANCE ANALYSIS ###\n")
    for algo in tqdm(algos, desc="Processing algorithms", position=0):
        # Plot the results
        for var in plot_vars.keys():
            # Get data
            data = np.array([results[algo][lvl]['performance'][var] for lvl in training])

            # Create DataFrame for seaborn
            df = pd.DataFrame({
                'Training Level': training,
                plot_vars[var]['ylabel']: data,
                'Algorithm': [algo] * len(training)
            })

            # Init the plot with seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df,
                x='Training Level',
                y=plot_vars[var]['ylabel'],
                hue='Algorithm',
                marker='X',
                s=150,
                ax=ax
            )

            # Add trend line
            sns.regplot(
                data=df,
                x='Training Level',
                y=plot_vars[var]['ylabel'],
                scatter=False,
                ci=None,
                line_kws={'linestyle':'--', 'alpha':0.7},
                ax=ax
            )

            # Styling
            plt.title(plot_vars[var]['title'], fontweight='bold')
            plt.xlabel(plot_vars[var]['xlabel'])
            plt.ylabel(plot_vars[var]['ylabel'])
            sns.despine(left=False, bottom=False)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save or show the plot
            if save_plot:
                plt.savefig(f"{root_dir}{var}/{algo}.png", dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

def performance_contour(results: dict, save_plot: bool = True, dpi: int = 600) -> None:
    """
    Plot contour maps of time-to-threshold for each algorithm and reward threshold.

    Parameters:
    -----------
    results: dict
        Full results dictionary.
    thresholds: list
        List of absolute reward thresholds to evaluate.
    save_plot: bool
        Whether to save the plot (shown if False).
    dpi: int
        Dots per inch for the plot.

    Returns:
    --------
    None
    """
    root_dir = 'plots/performance_contour/'
    algos = list(results.keys())
    training = np.array(list(results[algos[0]].keys()))

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    for algo in tqdm(algos, desc="Generating performance contours"):
        max_reward = min([max(res['Pbar']['reward']) for lvl, res in results[algo].items()])
        min_reward = max([min(res['Pbar']['reward']) for lvl, res in results[algo].items()])
        # Define thresholds
        base = (max_reward + min_reward)/2
        thresholds = np.linspace(base, max_reward, 5).astype(int)
        thresholds = sorted(thresholds, reverse=True)

        # Create a dataframe to store all data
        df_list = []
        for i, threshold in enumerate(thresholds):
            t_Xs = []
            for lvl in reversed(training):
                rewards = np.array(results[algo][lvl]['Pbar']['reward'])
                episodes = np.array(results[algo][lvl]['Pbar']['episode'])
                max_episode = max(episodes)
                indices = np.where(rewards >= threshold)[0]
                t_X = episodes[indices[0]] / max_episode if len(indices) > 0 else np.nan
                t_Xs.append(t_X)

            for j, (lvl, t_X) in enumerate(zip(training, t_Xs)):
                df_list.append({
                    'Training Level': lvl,
                    'Relative Time': t_X,
                    'Threshold': f'Reward ≥ {threshold}'
                })

        df = pd.DataFrame(df_list)

        # Create a figure with custom palette
        fig, ax = plt.subplots(figsize=(12, 8))
        custom_palette = sns.color_palette("husl", len(thresholds))

        # Plot lines for each threshold
        for i, threshold in enumerate(thresholds):
            threshold_data = df[df['Threshold'] == f'Reward ≥ {threshold}']
            sns.lineplot(
                data=threshold_data,
                x='Training Level',
                y='Relative Time',
                label=f'Reward ≥ {threshold}',
                color=custom_palette[i],
                linewidth=3,
                ax=ax
            )

            # Fill the area under the curve
            threshold_data = threshold_data.sort_values('Training Level')
            ax.fill_between(
                threshold_data['Training Level'],
                threshold_data['Relative Time'],
                0,
                alpha=0.3,
                color=custom_palette[i]
            )

        # Styling
        ax.set_title(f"{algo} - Time to Reach Reward Thresholds", fontweight='bold', fontsize=16)
        ax.set_xlabel("Pre-training Level [%]", fontsize=14)
        ax.set_ylabel("Relative Time to Threshold [-]", fontsize=14)
        sns.despine(left=False, bottom=False)
        plt.legend(title="Thresholds", title_fontsize=12, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{root_dir}{algo}_contour.png", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def transfer_gap_plot(results: dict, save_plot: bool = True, dpi: int = 600) -> None:
    """
    Plot pre- and post-transfer reward averages for each algorithm and pre-training level.

    Parameters:
    -----------
    results: dict
        Full results dictionary.
    save_plot: bool
        Whether to save the plot (shown if False).
    dpi: int
        Dots per inch for the plot.

    Returns:
    --------
    None
    """
    root_dir = 'plots/transfer_gap/'
    algos = list(results.keys())
    training = np.array(list(results[algos[0]].keys()))

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    for algo in tqdm(algos, desc="Generating transfer gap plots"):
        # Create dataframe for pre and post transfer data
        df_list = []

        for lvl in training:
            rewards = np.array(results[algo][lvl]['Pbar']['reward'])
            pre_transfer = np.mean(rewards[:10]) if len(rewards) >= 10 else np.nan
            post_transfer = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.nan

            df_list.append({
                'Training Level': lvl,
                'Reward': pre_transfer,
                'Stage': 'Before Transfer'
            })
            df_list.append({
                'Training Level': lvl,
                'Reward': post_transfer,
                'Stage': 'After Transfer'
            })

        df = pd.DataFrame(df_list)

        # Create plot with seaborn
        plt.figure(figsize=(12, 8))
        g = sns.lineplot(
            data=df,
            x='Training Level',
            y='Reward',
            hue='Stage',
            style='Stage',
            markers=['o', 'X'],
            dashes=[(None, None), (None, None)],
            markersize=12,
            linewidth=3,
            palette=['#3498db', '#e74c3c']
        )

        # Customize appearance
        plt.title(f"{algo} - Transfer Gap", fontweight='bold', fontsize=16)
        plt.xlabel("Pre-training Level [%]", fontsize=14)
        plt.ylabel("Avg Reward (10 episodes)", fontsize=14)
        sns.despine()
        plt.legend(title="", loc='best', frameon=True, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add the gap as shaded area
        for i in range(len(training)):
            before = df[(df['Stage'] == 'Before Transfer') & (df['Training Level'] == training[i])]['Reward'].values[0]
            after = df[(df['Stage'] == 'After Transfer') & (df['Training Level'] == training[i])]['Reward'].values[0]
            plt.fill_between([training[i]-0.5, training[i]+0.5], [before, before], [after, after],
                             alpha=0.2, color='gray')

        plt.tight_layout()
        if save_plot:
            plt.savefig(f"{root_dir}{algo}_transfer.png", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def training(results: dict, ctrl_vars: dict, save_plot: bool=True, dpi: int=600) -> None:
    """
    Plot the training runs for each algorithm.

    Parameters:
    -----------
    results: dict
        Full results dictionary.
    ctrl_vars: dict
        Control variables.
    save_plot: bool
        Whether to save the plot (shown if False).
    dpi: int
        Dots per inch for the plot.

    Returns:
    --------
    None
    """
    # Init root dir
    root_dir = 'plots/training/'
    # Init iterables
    algos = list(results.keys())
    training = np.array(list(results[algos[0]].keys()))
    plot_features = {
        'title': 'Reward vs Episode',
        'xlabel': 'Episode [-]',
        'ylabel': 'Reward [-]'
    }

    if save_plot:
        os.makedirs(root_dir, exist_ok=True)

    tqdm.write("\n### TRAINING PLOTS ###\n")
    for algo in tqdm(algos, desc="Processing algorithms", position=0):
        # Prepare data in DataFrame format for seaborn
        df_list = []
        for i, lvl in enumerate(training):
            for ep, reward in zip(results[algo][lvl]['Pbar']['episode'], results[algo][lvl]['Pbar']['reward']):
                df_list.append({
                    'Episode': ep,
                    'Reward': reward,
                    'Training Level': f"{lvl}%"
                })

        df = pd.DataFrame(df_list)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 9))

        # Plot lines for each training level with custom palette
        palette = sns.color_palette("husl", len(training))
        sns.lineplot(
            data=df,
            x='Episode',
            y='Reward',
            hue='Training Level',
            palette=palette,
            linewidth=2.5,
            alpha=0.8,
            ax=ax
        )

        # Add Pbar_max dashed horizontal line
        ax.axhline(
            y=ctrl_vars[algo]['Pbar_max'][0],
            color='black',
            linestyle='--',
            linewidth=2,
            label=r'$\bar{P}_{max,\,HiFi}$'
        )

        # Add t_max dashed vertical line
        ax.axvline(
            x=ctrl_vars[algo]['t_max'][0],
            color='black',
            linestyle='-.',
            linewidth=2,
            label=r'$t_{max,\,HiFi}$'
        )

        # Styling
        ax.set_title(f"{algo} - {plot_features['title']}", fontweight='bold', fontsize=16)
        ax.set_xlabel(plot_features['xlabel'], fontsize=14)
        ax.set_ylabel(plot_features['ylabel'], fontsize=14)

        # Move the training level legend outside
        training_handles, training_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        # Create two legends
        training_legend = ax.legend(
            handles=training_handles[:-2],
            labels=training_labels[:-2],
            title="Pre-training Level",
            loc='upper right',
            frameon=True,
            framealpha=0.9,
            fontsize=11
        )
        ax.add_artist(training_legend)

        # Add the threshold legend
        ax.legend(
            handles=training_handles[-2:],
            labels=[r'$\bar{P}_{max,\,HiFi}$', r'$t_{max,\,HiFi}$'],
            title="Thresholds",
            loc='lower left',
            frameon=True,
            framealpha=0.9,
            fontsize=11
        )

        sns.despine()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save or show the plot
        if save_plot:
            plt.savefig(f"{root_dir}{algo}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
