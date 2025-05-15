import argparse
from jsonargparse import CLI
import numpy as np
import os
import time
import csv
import datetime
from collections import defaultdict

# Local imports
import CR_LoFi.main as lofi
import CR_HiFi.main as hifi

def check_training_completed(log_path, target_timesteps):
    """Check if training reached the target number of timesteps."""
    if not os.path.exists(log_path):
        return False
        
    try:
        # Read the CSV log and get the max timesteps
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            max_timesteps = 0
            for row in reader:
                timesteps = int(row['timesteps'])
                max_timesteps = max(max_timesteps, timesteps)
        
        # Check if we reached the target
        return max_timesteps >= target_timesteps
    except Exception as e:
        print(f"Error checking log completion: {e}")
        return False

def write_timing_row(algo, pct, env_type, duration):
    csv_path = f"/scratch/amoec/ATC_RL/{algo}_ts.csv"
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['percentage', 'lofi-duration', 'hifi-duration'])
    
    # Read existing data to get any previous timing for this percentage
    existing_data = defaultdict(lambda: [None, None])
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pct_val = row['percentage']
                existing_data[pct_val] = [
                    float(row['lofi-duration']) if row['lofi-duration'] != '' else None,
                    float(row['hifi-duration']) if row['hifi-duration'] != '' else None
                ]
    
    # Update with new timing
    if env_type == 'lofi':
        existing_data[str(pct)][0] = duration
    else:  # hifi
        existing_data[str(pct)][1] = duration
    
    # Write all data back
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['percentage', 'lofi-duration', 'hifi-duration'])
        for pct_val, timings in existing_data.items():
            writer.writerow([pct_val] + timings)

def main(args):
    algo = args.algo
    runtime = args.runtime
    n_incr = args.n_incr
    window = args.window
    base_seed = args.seed
    config_path = f"CR_LoFi/atcenv/config/algos/{algo}.yaml"
    
    timeout = datetime.datetime.now() + datetime.timedelta(hours=runtime)
    
    if args.baseline:
        # Run 5 full baseline runs with different seeds.
        seeds = [base_seed + i + 1 for i in range(4)]
        # Use new csv file for baseline timings
        csv_path = f"/scratch/amoec/ATC_RL/baseline/{algo}_ts.csv"
        for seed in seeds:
            # Define folder paths for baseline runs
            hifi_path = f"/scratch/amoec/ATC_RL/baseline/HiFi-{algo}/run_{seed}"
            
            # Full training: HiFi
            hifi_folder = f"{hifi_path}/{algo}_full"
            hifi_log_path = f"{hifi_folder}/logs/results.csv"
            training_complete = check_training_completed(hifi_log_path, int(2e6))
            
            if not os.path.exists(hifi_folder) or not training_complete:
                restart = os.path.exists(hifi_folder) and not training_complete
                
                hifi_args = argparse.Namespace(
                    algorithm=algo,
                    train=True,
                    eval=False,
                    render=False,
                    pre_train='full',
                    window=window,
                    seed=seed,
                    baseline=True,
                    restart=False,
                    timeout=timeout
                )
                
                start_time = time.time()
                hifi.main(hifi_args)
                write_timing_row(algo, 'full', 'hifi', time.time() - start_time)
    else:
        # Create training increments
        pcts = np.round(np.linspace(0, 100, n_incr+1), 2)
        
        lofi_path = f"/scratch/amoec/ATC_RL/LoFi-{algo}"
        hifi_path = f"/scratch/amoec/ATC_RL/HiFi-{algo}"
        
        # Incremental training runs
        for pct in pcts:
            
            # LoFi training with percentage target
            lofi_folder = f"{lofi_path}/{algo}_{pct}"
            lofi_log_path = f"{lofi_folder}/logs/results.csv"
            target_timesteps = int((pct/100) * 3e6)
            training_complete = check_training_completed(lofi_log_path, target_timesteps)
            
            if not os.path.exists(lofi_folder) or not training_complete:
                restart = os.path.exists(lofi_folder) and not training_complete
                
                lofi_args = [
                    '--config', config_path,
                    '--pre_training', str(pct),
                    '--window', str(window),
                    '--algorithm', algo,
                    '--timeout', str(timeout),
                    '--train', 'true',
                    '--eval', 'false',
                    '--seed', str(base_seed),
                    '--restart', str(restart)
                ]
                
                start_time = time.time()
                CLI(lofi.main, as_positional=False, args=lofi_args)
                write_timing_row(algo, pct, 'lofi', time.time() - start_time)
            
            # HiFi training with pre-training percentage
            hifi_folder = f"{hifi_path}/{algo}_{pct}"
            hifi_log_path = f"{hifi_folder}/logs/results.csv"
            training_complete = check_training_completed(hifi_log_path, int(2e6))
            
            if not os.path.exists(hifi_folder) or not training_complete:
                restart = os.path.exists(hifi_folder) and not training_complete
                
                hifi_args = argparse.Namespace(
                    algorithm=algo,
                    timeout=timeout,
                    train=True,
                    eval=False,
                    render=False,
                    pre_train=str(pct),
                    window=window,
                    seed=base_seed,
                    restart=False,
                    baseline=False
                )
                
                start_time = time.time()
                hifi.main(hifi_args)
                write_timing_row(algo, pct, 'hifi', time.time() - start_time)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='RL algorithm to use')
    parser.add_argument('--runtime', type=float, required=True, help='Runtime in hours')
    parser.add_argument('--n_incr', type=int, required=True, help='Number of training increments')
    parser.add_argument('--window', type=int, required=True, help='Window size for moving average')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--baseline', action='store_true', help='Run baseline full training on multiple seeds')
    args = parser.parse_args()
    main(args)