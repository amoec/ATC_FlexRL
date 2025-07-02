import pandas as pd
import numpy as np
import warnings

def _assign_env(df: pd.DataFrame, pct: float):
    """
    Add an 'env' column: 0=LoFi, 1=HiFi, and return switch episode t_s.
    
    For pct==100.0 (pure LoFi): env=0 everywhere, t_s = last episode.
    For pct==0.0   (pure HiFi): env=1 everywhere, t_s = -1.
    """
    df = df.copy()
    
    # Base is all LoFi
    df['env'] = 0
    
    # Find and set the HiFi episodes
    mask_hifi = df['episode'] > pct* df['episode'].max() / 100
    df.loc[mask_hifi, 'env'] = 1
    
    # Set t_s (switch episode)
    if pct == 100.0:
        t_s = int(df['episode'].max())
    elif pct == 0.0:
        t_s = -1
    else:
        # Find the first episode where env switches to 1
        t_s = int(df.loc[df['env'] == 1, 'episode'].min())
    
    return df, t_s


def dropoff(aggregated_per_seed: dict, N: int) -> dict:
    """
    Transfer gap metrics: t_rec, tau, tau_rel, Pbar_s, Pbar_d, DeltaPbar_d,
    DeltaPbar_d_rel, DeltaPbar_star_d_rel.
    Metrics are averaged over seeds, and std dev is provided.
    """
    all_m_aggregated = {}
    for algo, pct_map_seeds in aggregated_per_seed.items():
        all_m_aggregated[algo] = {}
        for pct, seed_df_list in pct_map_seeds.items():
            if pct in (0.0, 100.0) or not seed_df_list:
                continue

            per_seed_metrics_list = []
            for seed_df in seed_df_list:
                # prepare per seed
                # Input seed_df has 'episode' and 'mean_reward' (which is actual reward for this seed)
                Pbar_seed = seed_df[['episode', 'mean_reward']].copy().rename(columns={'mean_reward': 'reward'})
                Pbar_seed, t_s = _assign_env(Pbar_seed, pct)
                
                if t_s == -1 and pct != 0.0: # Should not happen if _assign_env is correct for mixed runs
                    # This case means it's treated as pure HiFi by _assign_env, skip dropoff
                    # Or if Pbar_seed['env']==0 has no data
                    if not (Pbar_seed['env'] == 0).any():
                        warnings.warn(f"Algo {algo}, Pct {pct}: No LoFi data found for a seed, skipping dropoff calc for this seed.")
                        continue
                
                t_end = int(Pbar_seed['episode'].max())
                Pmin_seed, Pmax_seed = Pbar_seed['reward'].min(), Pbar_seed['reward'].max()
                
                # Ensure there are LoFi environment data points
                lofi_rewards = Pbar_seed.loc[Pbar_seed['env']==0, 'reward']
                if lofi_rewards.empty or len(lofi_rewards) <= N : # Ensure enough data for .tail(N+1).mean()
                    Pbar_s_seed = np.nan
                else:
                    Pbar_s_seed = float(lofi_rewards.tail(N+1).mean())
                # initialize metrics (no more _d_rel)
                current_seed_m = dict(
                    t_rec=np.nan, tau=np.nan, tau_rel=np.nan,
                    Pbar_s=np.nan,
                    Pbar_d=np.nan, DeltaPbar_d=np.nan,
                    DeltaPbar_d_rel=np.nan,
                    DeltaPbar_star=np.nan   # NEW: absolute step‐delta
                )

                if np.isnan(Pbar_s_seed): # Cannot proceed if Pbar_s is NaN
                    per_seed_metrics_list.append(current_seed_m)
                    continue

                # recovery
                rec_cond = ((Pbar_seed['env']==1) &
                       (Pbar_seed['reward'] >= Pbar_s_seed) &
                       (Pbar_seed['episode'] > t_s + N))
                if rec_cond.any():
                    t_rec_seed = int(Pbar_seed.loc[rec_cond, 'episode'].iloc[0])
                    tau_seed = t_rec_seed - t_s
                    current_seed_m.update(t_rec=t_rec_seed, tau=tau_seed, tau_rel=tau_seed/t_end if t_end else np.nan)

                    drop_cond = ((Pbar_seed['env']==1) &
                            (Pbar_seed['episode'] <= t_rec_seed) & # Use t_rec_seed
                            (Pbar_seed['episode'] > t_s + N/4))
                    if drop_cond.any():
                        Pbar_d_seed = float(Pbar_seed.loc[drop_cond, 'reward'].mean())
                        delta_seed = Pbar_d_seed - Pbar_s_seed
                        current_seed_m.update(
                            Pbar_d=Pbar_d_seed,
                            DeltaPbar_d=delta_seed,
                            DeltaPbar_d_rel=100*delta_seed/(Pbar_s_seed - Pmin_seed) if Pbar_s_seed!=Pmin_seed else np.nan,
                            DeltaPbar_star_d_rel=100*delta_seed/(Pmax_seed - Pmin_seed) if Pmax_seed!=Pmin_seed else np.nan
                        )
                ep = Pbar_seed['episode']
                rw = Pbar_seed['reward']
                pre_mask  = (ep > max(t_s-100,0)) & (ep <= t_s)
                post_mask = (ep > t_s) & (ep <= t_s+100)
                P_pre  = rw[pre_mask].mean()  if not rw[pre_mask].empty  else np.nan
                P_post = rw[post_mask].mean() if not rw[post_mask].empty else np.nan
                if pd.notna(P_pre) and pd.notna(P_post):
                    current_seed_m['DeltaPbar_star'] = P_post - P_pre
                per_seed_metrics_list.append(current_seed_m)
            
            if not per_seed_metrics_list:
                all_m_aggregated[algo][pct] = {} # No valid seed data
                continue

            # Aggregate metrics from all seeds for this (algo, pct)
            metrics_df = pd.DataFrame(per_seed_metrics_list)
            final_metrics_dict = {}
            for col in metrics_df.columns:
                final_metrics_dict[col + '_mean'] = metrics_df[col].mean()
                final_metrics_dict[col + '_std']  = metrics_df[col].std(ddof=0)
            all_m_aggregated[algo][pct] = final_metrics_dict
            
    return all_m_aggregated

def performance(aggregated_per_seed: dict,
                avg_times: dict, # avg_times[algo][pct]['lofi_duration_mean'/'_std']
                N: int) -> dict:
    """
    Compute performance metrics. The baseline is the 0.0% pre-training (pure HiFi) run.
    DeltaPbar: (Avg reward of final N episodes of current pct run) - (Avg reward of final N episodes of 0.0% HiFi run for the same algo).
    t_XO metrics are set to NaN.
    T_tot is the total average duration for a given pct.
    T_tot_rel is T_tot normalized by the total average duration of the 0.0% HiFi baseline.
    Metrics are averaged over seeds, and std dev is provided.
    Returns nested dict[algo][pct] -> metrics_with_mean_std, excluding pct=0.0 and pct=100.0.
    """
    perf_aggregated = {}

    for algo, pct_map_seeds in aggregated_per_seed.items():
        perf_aggregated[algo] = {}

        # --- 1. Establish Baseline for the current algorithm (from 0.0% HiFi runs) ---
        if 0.0 not in pct_map_seeds or not pct_map_seeds[0.0]:
            warnings.warn(f"Algo {algo}: Missing 0.0% HiFi data (pct=0.0) for baseline. Skipping performance metrics for this algo.")
            continue
        
        baseline_hifi_seed_dfs = pct_map_seeds[0.0]
        
        T_hifi_baseline_duration_mean = np.nan
        if algo in avg_times and 0.0 in avg_times[algo] and 'hifi_duration_mean' in avg_times[algo][0.0]:
            T_hifi_baseline_duration_mean = avg_times[algo][0.0]['hifi_duration_mean']
        else:
            warnings.warn(f"Algo {algo}: Missing timing data for 0.0% HiFi baseline (avg_times[algo][0.0]['hifi_duration_mean']). T_tot_rel will be NaN.")

        # 1) baseline final‐100 mean
        seed_final100_baseline = []
        for base_hifi_seed_df in baseline_hifi_seed_dfs:
            df0 = base_hifi_seed_df[['episode','mean_reward']].rename(columns={'mean_reward':'reward'})
            if len(df0) >= 100:
                seed_final100_baseline.append(df0['reward'].tail(100).mean())
            else:
                seed_final100_baseline.append(df0['reward'].mean())
                
        P_hifi_baseline_final100_mean = np.nanmean(seed_final100_baseline)
        

        # --- 2. Calculate metrics for each pct level for the current algo ---
        for pct, seed_df_list in pct_map_seeds.items():
            if not seed_df_list:
                if pct != 0.0 and pct != 100.0: # Only add empty dict for mixed pct if no data
                    perf_aggregated[algo][pct] = {}
                continue # Skip further processing for this pct

            # Calculate T_tot and T_tot_rel for the current (algo, pct)
            T_total_for_current_pct = np.nan
            T_total_rel_for_current_pct = np.nan # This is the raw ratio T_current / T_baseline

            if algo in avg_times and pct in avg_times[algo]:
                lofi_dur_mean = avg_times[algo][pct].get('lofi_duration_mean', 0.0) 
                hifi_dur_mean = avg_times[algo][pct].get('hifi_duration_mean', 0.0) 
                
                if pd.isna(avg_times[algo][pct].get('lofi_duration_mean')) and pct != 0.0 : lofi_dur_mean = np.nan
                if pd.isna(avg_times[algo][pct].get('hifi_duration_mean')) and pct != 100.0 : hifi_dur_mean = np.nan

                if pct == 0.0: 
                    T_total_for_current_pct = hifi_dur_mean
                elif pct == 100.0: 
                    T_total_for_current_pct = lofi_dur_mean
                else: 
                    if pd.notna(lofi_dur_mean) and pd.notna(hifi_dur_mean):
                        T_total_for_current_pct = lofi_dur_mean + hifi_dur_mean
                    else:
                        T_total_for_current_pct = np.nan
                
                if pd.notna(T_total_for_current_pct) and pd.notna(T_hifi_baseline_duration_mean) and T_hifi_baseline_duration_mean > 0:
                    T_total_rel_for_current_pct = T_total_for_current_pct / T_hifi_baseline_duration_mean
            else:
                warnings.warn(f"Algo {algo}, Pct {pct}: Missing timing data in avg_times. T_tot and T_tot_rel will be NaN.")

            per_seed_metrics_list = []
            # Pre-calculate the transformed T_tot_rel for this (algo, pct)
            _T_tot_rel_transformed_for_pct = (T_total_rel_for_current_pct-1) * 100 if pd.notna(T_total_rel_for_current_pct) else np.nan

            for seed_df in seed_df_list:
                dfc = seed_df[['episode','mean_reward']].rename(columns={'mean_reward':'reward'})
                # last-100 window
                if len(dfc) >= 100:
                    P_curr100 = dfc['reward'].tail(100).mean()
                else:
                    P_curr100 = dfc['reward'].mean()
                deltaP_seed = P_curr100 - P_hifi_baseline_final100_mean
                
                current_seed_metrics = {
                    'DeltaPbar': deltaP_seed,        # NEW absolute final‐100 delta
                    'T_tot':    T_total_for_current_pct,
                    'T_tot_rel':_T_tot_rel_transformed_for_pct
                }
                per_seed_metrics_list.append(current_seed_metrics)
                
            # --- 3. Aggregate metrics from all seeds for this (algo, pct) ---
            # If seed_df_list was not empty, per_seed_metrics_list will not be empty.
            metrics_df = pd.DataFrame(per_seed_metrics_list)
            final_metrics_dict = {}
            for col_name in metrics_df.columns: 
                final_metrics_dict[col_name + '_mean'] = metrics_df[col_name].mean()
                final_metrics_dict[col_name + '_std']  = metrics_df[col_name].std(ddof=0)
            
            # Only add to the output if it's a mixed percentage (not 0.0% or 100.0%)
            if pct != 0.0 and pct != 100.0:
                perf_aggregated[algo][pct] = final_metrics_dict
            
    return perf_aggregated

def performance_contour(perf_metrics: dict) -> dict:
    """
    Convert perf_metrics to DataFrames keyed by algo, containing only normalized mean metrics.
    Normalized metrics are assumed to have '_rel_mean' suffix.
    """
    contour_data = {}
    # Define the base names of relative metrics you want in the contour plot
    # These should correspond to keys in the dicts from the performance function, before _mean/_std
    relative_metric_bases = ['DeltaPbar_rel', 't_XO_rel', 'T_tot_rel'] 

    for algo, pm_dict in perf_metrics.items():
        if not pm_dict: continue
        # Create a DataFrame from [pct] -> {metric_name_mean: value, metric_name_std: value}
        df = pd.DataFrame.from_dict(pm_dict, orient='index')
        
        # Select only the '_mean' versions of the relative metrics
        cols_to_select = [base + '_mean' for base in relative_metric_bases]
        
        # Filter out columns that might not exist (e.g., if a metric was NaN for all seeds)
        existing_cols_to_select = [col for col in cols_to_select if col in df.columns]
        
        if not existing_cols_to_select:
            warnings.warn(f"Algo {algo}: No normalized metrics found for contour plot.")
            contour_data[algo] = pd.DataFrame() # Empty DataFrame
            continue
            
        df_filtered = df[existing_cols_to_select]
        # Optional: Rename columns for prettier display in heatmap, e.g., remove '_mean' or '_rel_mean'
        # df_filtered.columns = [col.replace('_rel_mean', ' (Rel)') for col in existing_cols_to_select]
        contour_data[algo] = df_filtered
    return contour_data

def calculate_time_to_threshold(aggregated_per_seed: dict,
                                num_thresholds: int, # Changed from reward_thresholds
                                max_ep_for_run: int
                               ) -> dict:
    """
    Calculates the mean relative time (episode/total_episodes) to reach dynamically determined reward thresholds.
    Thresholds are generated based on the overall maximum reward found in the data and num_thresholds.
    The results are structured for a stacked area plot, meaning it returns the height of each layer.

    Parameters:
    -----------
    aggregated_per_seed: dict
        Data structure: results[algo][pct] -> list_of_seed_dfs.
        Each seed_df has 'episode' and 'mean_reward'.
    num_thresholds: int
        The number of reward thresholds to generate.
    max_ep_for_run: int
        A fallback for total episodes if a specific run's data is empty or lacks episode info.

    Returns:
    --------
    dict:
        output_data[algo] -> pd.DataFrame
        The DataFrame has pre-training percentages as index, threshold labels as columns,
        and values represent the mean height of the stack layer for that threshold.
    """
    output_data = {}

    if num_thresholds <= 0:
        warnings.warn("num_thresholds must be positive. Returning empty data for time_to_threshold.")
        return output_data

    # --- 1. Determine overall maximum reward to base thresholds on ---
    P_max_overall = -np.inf
    found_any_data = False
    for algo, pct_map_seeds in aggregated_per_seed.items():
        for pct, seed_df_list in pct_map_seeds.items():
            if not seed_df_list:
                continue
            for seed_df in seed_df_list:
                if not seed_df.empty and 'mean_reward' in seed_df.columns and not seed_df['mean_reward'].empty:
                    found_any_data = True
                    current_max_reward = seed_df['mean_reward'].max()
                    if pd.notna(current_max_reward) and current_max_reward > P_max_overall:
                        P_max_overall = current_max_reward
    
    if not found_any_data or pd.isna(P_max_overall) or P_max_overall == -np.inf :
        warnings.warn("No valid reward data found in aggregated_per_seed to determine P_max_overall. Skipping time_to_threshold calculation.")
        # Initialize with empty DataFrames for all algos if no thresholds can be made
        for algo in aggregated_per_seed.keys():
            pct_levels_for_algo = sorted(aggregated_per_seed[algo].keys())
            output_data[algo] = pd.DataFrame(index=pct_levels_for_algo, columns=[])
        return output_data

    # --- 2. Generate reward thresholds ---
    # Thresholds: P_max_overall, P_max_overall-1, ..., P_max_overall-(num_thresholds-1)
    generated_thresholds_raw = [P_max_overall - i for i in range(num_thresholds)]
    # Sorted from "easiest" (numerically smallest) to "hardest" for stack plot logic
    sorted_thresholds = sorted(list(set(generated_thresholds_raw))) 
    
    if not sorted_thresholds:
        warnings.warn("Generated no valid thresholds (P_max_overall might be problematic or num_thresholds too low). Skipping time_to_threshold calculation.")
        for algo in aggregated_per_seed.keys():
            pct_levels_for_algo = sorted(aggregated_per_seed[algo].keys())
            output_data[algo] = pd.DataFrame(index=pct_levels_for_algo, columns=[])
        return output_data
        
    threshold_labels = [f"Reward ≥ {th:.1f}" for th in sorted_thresholds]

    # --- 3. Calculate metrics for each algo/pct ---
    for algo, pct_map_seeds in aggregated_per_seed.items():
        algo_pct_data = {} 
        pct_levels_for_algo = sorted(pct_map_seeds.keys())

        for pct in pct_levels_for_algo:
            seed_df_list = pct_map_seeds.get(pct)
            
            if not seed_df_list: # No data for this specific pct, fill with NaNs for this pct
                algo_pct_data[pct] = dict(zip(threshold_labels, [np.nan] * len(threshold_labels)))
                continue

            all_seeds_cumulative_rts_for_pct = [] 

            for seed_df in seed_df_list:
                if seed_df.empty or 'episode' not in seed_df.columns or 'mean_reward' not in seed_df.columns:
                    all_seeds_cumulative_rts_for_pct.append([np.nan] * len(sorted_thresholds))
                    continue

                rewards = seed_df['mean_reward'].values
                episodes = seed_df['episode'].values
                
                current_run_total_episodes = np.nan
                if episodes.size > 0 : # Check if episodes array is not empty
                    current_run_total_episodes = episodes.max()

                if pd.isna(current_run_total_episodes) or current_run_total_episodes < 0:
                    current_run_total_episodes = max_ep_for_run
                
                if current_run_total_episodes == 0:
                    ep_0_reward = np.nan
                    if episodes.size > 0 and episodes[0] == 0 and rewards.size > 0:
                        ep_0_reward = rewards[0]
                    
                    seed_cumulative_rts_at_ep0 = []
                    for r_thresh in sorted_thresholds:
                        if pd.notna(ep_0_reward) and ep_0_reward >= r_thresh:
                            seed_cumulative_rts_at_ep0.append(0.0)
                        else:
                            seed_cumulative_rts_at_ep0.append(np.nan)
                    all_seeds_cumulative_rts_for_pct.append(seed_cumulative_rts_at_ep0)
                    continue

                seed_cumulative_rts = []
                for r_thresh in sorted_thresholds:
                    met_indices = np.where(rewards >= r_thresh)[0]
                    ep_at_threshold = np.nan
                    if len(met_indices) > 0:
                        first_met_idx = met_indices[0]
                        ep_at_threshold = episodes[first_met_idx]
                    
                    relative_time = np.nan
                    if pd.notna(ep_at_threshold) and current_run_total_episodes > 0 :
                        relative_time = ep_at_threshold / current_run_total_episodes
                    elif pd.notna(ep_at_threshold) and current_run_total_episodes == 0 and ep_at_threshold == 0:
                        relative_time = 0.0
                        
                    seed_cumulative_rts.append(relative_time)
                all_seeds_cumulative_rts_for_pct.append(seed_cumulative_rts)

            if not all_seeds_cumulative_rts_for_pct: 
                mean_cumulative_rts_for_pct = [np.nan] * len(sorted_thresholds)
            else:
                np_all_seeds_rts = np.array(all_seeds_cumulative_rts_for_pct, dtype=float)
                mean_cumulative_rts_for_pct = np.nanmean(np_all_seeds_rts, axis=0).tolist()
            
            stackplot_layers_for_pct = []
            previous_mean_cumulative_rt = 0.0
            for i in range(len(mean_cumulative_rts_for_pct)):
                current_mean_cumulative_rt = mean_cumulative_rts_for_pct[i]
                layer_height = np.nan

                if pd.notna(current_mean_cumulative_rt):
                    effective_previous_rt = previous_mean_cumulative_rt if pd.notna(previous_mean_cumulative_rt) else 0.0
                    layer_height = current_mean_cumulative_rt - effective_previous_rt
                    
                    if layer_height < 0: 
                        warnings.warn(
                            f"Algo {algo}, Pct {pct}: Negative layer height ({layer_height:.3f}) for '{threshold_labels[i]}' "
                            f"(current_cumulative_rt: {current_mean_cumulative_rt:.3f}, prev_cumulative_rt: {effective_previous_rt:.3f}). Setting to 0."
                        )
                        layer_height = 0.0 
                
                stackplot_layers_for_pct.append(layer_height)
                
                if pd.notna(current_mean_cumulative_rt):
                    previous_mean_cumulative_rt = current_mean_cumulative_rt

            algo_pct_data[pct] = dict(zip(threshold_labels, stackplot_layers_for_pct))

        if algo_pct_data:
            df_algo = pd.DataFrame.from_dict(algo_pct_data, orient='index')
            df_algo = df_algo.reindex(index=pct_levels_for_algo, columns=threshold_labels)
            output_data[algo] = df_algo
        else: 
            output_data[algo] = pd.DataFrame(index=pct_levels_for_algo, columns=threshold_labels)
            
    return output_data

def transfer_gap(perf_metrics: dict) -> dict:
    """
    Extract DeltaPbar_mean and DeltaPbar_std for each algo/pct.
    """
    gap_data = {}
    for algo, pm_dict in perf_metrics.items():
        gap_data[algo] = {}
        for pct, metrics_values in pm_dict.items():
            gap_data[algo][pct] = {
                'mean': metrics_values.get('DeltaPbar_mean', np.nan),
                'std': metrics_values.get('DeltaPbar_std', np.nan)
            }
    return gap_data
