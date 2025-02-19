import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

def kalman_filter(data, process_variance=1e-5, measurement_variance=1):
    estimates = np.zeros(len(data))
    posteri_estimate = data[0]
    posteri_error_estimate = 1.0
    for i, measurement in enumerate(data):
        priori_estimate = posteri_estimate
        priori_error_estimate = posteri_error_estimate + process_variance
        kalman_gain = priori_error_estimate / (priori_error_estimate + measurement_variance)
        posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        estimates[i] = posteri_estimate
    return estimates

def find_target(selected_algo, window=100, show_plot=True):
    '''
    Returns the target training performance, taken from the average of the final 250 episodes of the full training run
    '''
    with open(f'exp/LoFi-{selected_algo}/{selected_algo}_full/logs/results.csv', 'r') as f:
        lofi_results = pd.read_csv(f)
    
    with open(f'exp/HiFi-{selected_algo}/{selected_algo}_full/logs/results.csv', 'r') as f:
        hifi_results = pd.read_csv(f)
    
    lofi_cleaned = lofi_results.groupby('episode').sum().drop(columns=['timesteps'])
    hifi_cleaned = hifi_results.groupby('episode').sum().drop(columns=['timesteps'])
    
    lofi_cleaned = kalman_filter(lofi_cleaned['reward'].values)
    hifi_cleaned = kalman_filter(hifi_cleaned['reward'].values)

    
    lofi_target = np.mean(lofi_cleaned[-window:])
    hifi_target = np.mean(hifi_cleaned[-window:])
    
    plt.figure(figsize=(10, 6))
    plt.plot(lofi_cleaned, label='LoFi', color='blue')
    plt.plot(hifi_cleaned, label='HiFi', color='orange')
    plt.axhline(lofi_target, linestyle='-.', color='grey', alpha=0.75)
    plt.axhline(hifi_target, linestyle='--', color='grey', alpha=0.75)
    plt.title(f'{selected_algo} training performance')
    plt.legend()
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(f"plots/{selected_algo}_benchmark.png", dpi=300)
    
    return lofi_target, hifi_target

def main(selected_algo, smoothing_choice, show_target=False, show_plot=False, save=True):
    # Initialize result dataframes dict
    LoFi_results = dict.fromkeys(training_types)
    HiFi_results = dict.fromkeys(training_types)

    # Cylce through each training type
    for training in training_types:
        # LoFi results
        with open(f'exp/LoFi-{selected_algo}/{selected_algo}_{training}/logs/results.csv', 'r') as f:
            LoFi_results[training] = pd.read_csv(f)
        # HiFi results
        with open(f'exp/HiFi-{selected_algo}/{selected_algo}_{training}/logs/results.csv', 'r') as f:
            HiFi_results[training] = pd.read_csv(f)
    print(f'{selected_algo} results loaded, starting to clean data...')

    # Aggregate per episode results for each training type
    LoFi_cleaned = dict.fromkeys(training_types)
    HiFi_cleaned = dict.fromkeys(training_types)

    # Sum all rewards together were the episode number is the same, remove timestep column
    for training in training_types:
        LoFi_cleaned[training] = LoFi_results[training].groupby('episode').sum().drop(columns=['timesteps'])
        HiFi_cleaned[training] = HiFi_results[training].groupby('episode').sum().drop(columns=['timesteps'])

    if smoothing_choice == 1:
        window = int(input("Enter moving average window size: "))
    elif smoothing_choice == 2:
        pass
    else:
        print("Invalid selection. Defaulting to moving average with window = 250")
        smoothing_choice = 1
        window = 250
        
    if show_target:
        lofi_target, hifi_target = find_target(selected_algo)
        print(f"LoFi target: {lofi_target}\nHiFi target: {hifi_target}")
    
    # Plot LoFi and HiFi results one after another on the same line for each training type using the selected smoothing method
    plt.figure(figsize=(10, 6))

    # training_slice = np.linspace(0, 100, 5)[1:]
    training_slice = training_types[1:]
    cmap = plt.get_cmap('viridis')

    for idx, training in enumerate(training_slice):
        training = str(training)
        color = cmap(idx / len(training_slice))
        
        if smoothing_choice == 1:
            data_lofi = LoFi_cleaned[training]['reward'].rolling(window=window).mean()
            data_hifi = HiFi_cleaned[training]['reward'].rolling(window=window).mean()
        elif smoothing_choice == 2:
            data_lofi = kalman_filter(LoFi_cleaned[training]['reward'].values)
            data_hifi = kalman_filter(HiFi_cleaned[training]['reward'].values)
        
        data = np.hstack((data_lofi, data_hifi))
        
        # Add crossover markers for going from LoFi to HiFi
        final_episode = LoFi_cleaned[training].index[-1]
        
        plt.plot(data, label=f'{training}%', color=color, zorder=1)
        plt.scatter(final_episode, data[final_episode], marker='x', color='red', s=50, zorder=4)
        
        # Annotate with a padded white box
        plt.text(final_episode, data[final_episode], f' {training}% ', fontsize=12, zorder=3,
                bbox=dict(facecolor='white', alpha=1, edgecolor='black', pad=1))
        
        if show_target:
            plt.axhline(lofi_target, linestyle='-.', color='grey', alpha=0.75)
            plt.axhline(hifi_target, linestyle='--', color='grey', alpha=0.75)
        
    plt.title(f'{selected_algo} performance across lofi training percentages')
    plt.legend()
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    if save:
        plt.savefig(f"plots/{selected_algo}_training.png", dpi=300)
    
# Algorithm types
algos = ['A2C', 'PPO', 'SAC', 'DDPG', 'TD3']

# Training types
training_types = np.hstack((np.array(['full']), np.linspace(0, 100.0, 21)[1:]))

# TODO: Remove this
training_types = training_types[:-2]

if __name__=='__main__':
    selected_algo = int(input(f"Select from the following algorithms:\n1. A2C\n2. PPO\n3. SAC\n4. DDPG\n5. TD3\n")) - 1 # 0-indexed
    selected_algo = algos[selected_algo]

    smoothing_choice = int(input("Select smoothing method:\n1. Moving Average\n2. Kalman Filter\n"))
    
    main(selected_algo, smoothing_choice, show_target=True, show_plot=True, save=False)
