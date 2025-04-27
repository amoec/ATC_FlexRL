import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from typing import Callable

# Local imports
from common import metrics, plot
from common.filters import moving_avg, kalman

def main(N: int=50, save_plot: bool=True, save_metrics: bool=True) -> None:
    '''
    Perform full analysis on the results of the training runs.
    
    Parameters:
    -----------
    N: int
        Moving avg window (default=250 ep).
    save_plot: bool
        Whether to save the plots.
    
    save_metrics: bool
        Whether to save the metrics.
        
    Returns:
    --------
    None
    '''
    algos = ['A2C', 'PPO', 'SAC', 'TD3', 'DDPG']
    
    # Find and load all the complete data files. 21 folders for LoFi and HiFi.
    root_dir = 'exp/'
    
    # Define the training array
    training = np.linspace(0, 100, 21)[1:]
    
    # Init overall results dict
    results = {}
    
    # Init overall baseline vars
    ctrl = {}
    
    tqdm.write("\n### DATA POST-PROCESSING ###\n")
    # Create progress bar for algorithms
    for algo in tqdm(algos, desc="Processing algorithms", position=0):
        # Pretraining result locations
        dirpath_lofi = f"{root_dir}LoFi-{algo}/"
        dirpath_hifi = f"{root_dir}HiFi-{algo}/"
        
        # Baseline result locations
        dirpath_lofi_base = f"{root_dir}baseline/LoFi-{algo}/"
        dirpath_hifi_base = f"{root_dir}baseline/HiFi-{algo}/"
        # Extract baseline subfolders if they exist
        full_baseline = os.path.exists(dirpath_lofi_base) & os.path.exists(dirpath_hifi_base)
        
        if full_baseline:
            subfolders_hifi = [os.path.join(dirpath_hifi_base, f) for f in os.listdir(dirpath_hifi_base) if f.startswith("run_")]
        
        if len(os.listdir(dirpath_lofi)) and len(os.listdir(dirpath_hifi)) == 21:
            # Init rewards dict
            results[algo] = {lvl: {} for lvl in training}
            
            # Read training timestamps file
            ts = pd.read_csv(f"{root_dir}{algo}_ts.csv")
            
            # Get baseline data            
            baseline_vars = {
                'Pbar_min': [],
                'Pbar_max': [],
                'T_max': [],
                't_max': [],
                't_end': []
            }
            
            base_hifi = pd.read_csv(f"{dirpath_hifi}{algo}_full/logs/results.csv")
            
            # Aggregate per-episode rewards
            base_hifi = base_hifi.groupby('episode').sum().reset_index().drop(columns=['timesteps'])
            
            # Apply smoothing
            base_hifi = kalman(base_hifi, 'reward')
            base_hifi = moving_avg(base_hifi, 'reward', window=N)
            
            max_cond = base_hifi['episode'] >= base_hifi['episode'].iloc[-1]*0.05
            baseline_vars['Pbar_min'].append(base_hifi['reward'].min())
            baseline_vars['Pbar_max'].append(base_hifi.loc[max_cond, 'reward'].max())
            
            # Add time to max reward
            T_hifi = ts.loc[ts['percentage'] == 'full', 'hifi-duration'].iloc[0]
            ep_max = base_hifi['episode'].iloc[base_hifi.loc[max_cond, 'reward'].idxmax()]
            t_end = base_hifi['episode'].iloc[-1]
            baseline_vars['T_max'].append(ep_max * T_hifi / t_end)
            baseline_vars['t_max'].append(ep_max)
            baseline_vars['t_end'].append(t_end)
            # print(f"ALGO: {algo}")
            # print(f"Max reward: {base_hifi['reward'].max()}")
            # print(f"Rel max episode: {ep_max/t_end}")
            # print(f"End episode: {t_end}")
            # print(f"Max reward episode: {ep_max}")
                        
            if full_baseline:
                for i, hifi in enumerate(subfolders_hifi):
                    base_hifi = pd.read_csv(f"{hifi}/logs/results.csv")
                    
                    # Aggregate per-episode rewards
                    base_hifi = base_hifi.groupby('episode').sum().reset_index().drop(columns=['timesteps'])

                    # Apply smoothing
                    base_hifi = kalman(base_hifi, 'reward')
                    base_hifi = moving_avg(base_hifi, 'reward', window=N)
                    
                    max_cond = base_hifi['episode'] >= base_hifi['episode'].iloc[-1]*0.05
                    baseline_vars['Pbar_min'].append(base_hifi['reward'].min())
                    baseline_vars['Pbar_max'].append(base_hifi.loc[max_cond, 'reward'].max())
                    
                    # Store the episode number of the max reward
                    baseline_vars['t_max'].append(base_hifi['episode'].iloc[base_hifi.loc[max_cond, 'reward'].idxmax()])
                    
                    # Store the end episode number
                    baseline_vars['t_end'].append(base_hifi['episode'].iloc[-1])
                    
                # Add time to max reward
                ts_base = pd.read_csv(f"{root_dir}baseline/{algo}_ts.csv")
                T_hifi = ts_base['hifi-duration']
                T_max = np.multiply(baseline_vars['t_max'], T_hifi/np.array(baseline_vars['t_end'])) # [s]
                
                # Calculate the mean and std
                Pbar_min = np.mean(baseline_vars['Pbar_min'])
                Pbar_max = np.mean(baseline_vars['Pbar_max'])
                std_min = np.std(baseline_vars['Pbar_min'])
                std_max = np.std(baseline_vars['Pbar_max'])
                baseline_vars['Pbar_min'] = [Pbar_min, std_min]
                baseline_vars['Pbar_max'] = [Pbar_max, std_max]
                baseline_vars['T_max'] = [np.mean(T_max), np.std(T_max)]
                baseline_vars['t_max'] = [np.mean(baseline_vars['t_max']), np.std(baseline_vars['t_max'])]
            else:
                # Calculate the mean and std (only one run)
                baseline_vars['Pbar_min'].append(0)
                baseline_vars['Pbar_max'].append(0)
                baseline_vars['T_max'].append(0)
                baseline_vars['t_max'].append(0)
            
            # Store the baseline vars
            ctrl[algo] = baseline_vars
            
            # Create nested progress bar for training levels
            for lvl in tqdm(training, desc=f"Processing training levels for {algo}", position=1, leave=False):
                
                # Step 1: Read data
                df_lofi = pd.read_csv(f"{dirpath_lofi}{algo}_{lvl}/logs/results.csv")
                df_hifi = pd.read_csv(f"{dirpath_hifi}{algo}_{lvl}/logs/results.csv")
                
                # Step 2: Aggregate rewards
                df_lofi = df_lofi.groupby('episode').sum().reset_index()
                df_lofi = df_lofi.drop(columns=['timesteps'])
                df_hifi = df_hifi.groupby('episode').sum().reset_index()
                df_hifi = df_hifi.drop(columns=['timesteps'])
                
                # Add type flag
                df_lofi['env'] = 0
                df_hifi['env'] = 1
                
                # Adjust hifi episode numbers to continue from where lofi ends
                max_lofi_episode = df_lofi['episode'].max()
                df_hifi['episode'] = df_hifi['episode'] + max_lofi_episode + 1
                
                df = pd.concat([df_lofi, df_hifi], ignore_index=True)
                
                # Concatenate the data into single df
                results[algo][lvl]['raw'] = df
                
                # Step 3: Apply smoothing
                Pbar_df = kalman(df, 'reward')
                Pbar_df = moving_avg(Pbar_df, 'reward', window=N)
                results[algo][lvl]['Pbar'] = Pbar_df
                
                # Step 4: Add timestamps
                results[algo][lvl]['time'] = {
                    'lofi': ts.loc[ts["percentage"] == str(lvl), "lofi-duration"].iloc[0],
                    'hifi': ts.loc[ts["percentage"]  == str(lvl), "hifi-duration"].iloc[0]
                }
        
                # Step 5: Calculate metrics
                results[algo][lvl]['dropoff'] = metrics.dropoff(results[algo][lvl], N)
                results[algo][lvl]['performance'] = metrics.performance(results[algo][lvl], baseline_vars, N)

    tqdm.write("All data processed...")
    # Save the results
    if save_metrics:
        tqdm.write("Saving the results...")
        with open('data/results.pkl', 'wb') as f:
            pickle.dump(results, f)
        tqdm.write("Saved...")
    else:
        tqdm.write("WARNING: Results not saved...")
    
    
    # Plot the results
    plot.dropoff(results, save_plot)
    plot.performance_contour(results, save_plot)
    plot.transfer_gap_plot(results, save_plot)
    plot.performance(results, save_plot)
    plot.training(results, ctrl, save_plot)
        
if __name__ == '__main__':
    main()