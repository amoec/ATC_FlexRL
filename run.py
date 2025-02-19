import argparse
from jsonargparse import CLI
import numpy as np
import os
import time
import csv
from collections import defaultdict

# Local imports
import atcenv_gym.main as lofi
import SecCREnv.main as hifi

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
    n_incr = args.n_incr
    window = args.window
    seed = args.seed
    config_path = f"atcenv_gym/atcenv/config/algos/{algo}.yaml"
    
    # Create training increments
    pcts = np.round(np.linspace(0, 100, n_incr+1), 2)
    
    lofi_path = f"/scratch/amoec/ATC_RL/LoFi-{algo}"
    hifi_path = f"/scratch/amoec/ATC_RL/HiFi-{algo}"
    
    # Full training
    if not os.path.exists(f"{lofi_path}/{algo}_full"):
        lofi_args = [
            '--config', config_path,
            '--pre_training', 'full',
            '--window', str(window),
            '--algorithm', algo,
            '--train', 'true',
            '--eval', 'false',
            '--seed', str(seed)
        ]
        start_time = time.time()
        CLI(lofi.main, as_positional=False, args=lofi_args)
        write_timing_row(algo, 'full', 'lofi', time.time() - start_time)
    
    if not os.path.exists(f"{hifi_path}/{algo}_full"):
        hifi_args = argparse.Namespace(
            algorithm=algo,
            train=True,
            eval=False,
            render=False,
            pre_train='full',
            window=window,
            seed=seed,
        )
        start_time = time.time()
        hifi.main(hifi_args)
        write_timing_row(algo, 'full', 'hifi', time.time() - start_time)
    
    for pct in pcts:
        if pct == 0:
            continue
        
        if not os.path.exists(f"{lofi_path}/{algo}_{pct}"):
            lofi_args = [
                '--config', config_path,
                '--pre_training', str(pct),
                '--window', str(window),
                '--algorithm', algo,
                '--train', 'true',
                '--eval', 'false',
                '--seed', str(seed)
            ]
            start_time = time.time()
            CLI(lofi.main, as_positional=False, args=lofi_args)
            write_timing_row(algo, pct, 'lofi', time.time() - start_time)
        
        if not os.path.exists(f"{hifi_path}/{algo}_{pct}"):
            hifi_args = argparse.Namespace(
                algorithm=algo,
                train=True,
                eval=False,
                render=False,
                pre_train=str(pct),
                window=window,
                seed=seed,
            )
            start_time = time.time()
            hifi.main(hifi_args)
            write_timing_row(algo, pct, 'hifi', time.time() - start_time)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='RL algorithm to use')
    parser.add_argument('--n_incr', type=int, required=True, help='Number of training increments')
    parser.add_argument('--window', type=int, required=True, help='Window size for moving average')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()
    main(args)