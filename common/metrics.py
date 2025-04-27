import pandas as pd
import numpy as np

def dropoff(dfs: dict, N: int) -> dict:
    '''
    Calculate the dropoff metrics from env switch.
    
    Parameters:
    -----------
    dfs: dict
        Raw and moving avg dataframes.
    N: int
        Moving avg window.
    
    Returns:
    --------
    dict
        Dropoff metrics.
    '''
    # Init metrics dict
    metrics = {
        't_rec': np.nan,
        'tau': np.nan,
        'tau_rel': np.nan,
        'Pbar_s': np.nan,
        'Pbar_d': np.nan,
        'DeltaPbar_d': np.nan,
        'DeltaPbar_d_rel': np.nan,
        'DeltaPbar*_d_rel': np.nan
    }
    
    # Extract the dataframes
    P = dfs['raw']
    Pbar = dfs['Pbar']
    
    # Useful constants
    t_s = Pbar.loc[Pbar['env'] == 0, 'episode'].iloc[-1] # Switch episode
    t_end = Pbar['episode'].iloc[-1] # End episode
    Pbar_min = Pbar['reward'].min()# Min reward
    Pbar_max = Pbar['reward'].max() # Max reward
    
    # Find the switch peak
    Pbar_s = Pbar.loc[Pbar['env'] == 0, 'reward'].iloc[-(N+1):].mean()
    
    # Update the metrics dict
    metrics['Pbar_s'] = Pbar_s
    
    # Find the recovery episode
    rec_cond = (Pbar['env'] == 1) & (Pbar['reward'] >= Pbar_s) & (Pbar['episode'] > t_s + N/4)
    
    # Protect against irrecoverable cases
    if rec_cond.any():
        t_rec = Pbar.loc[rec_cond, 'episode'].iloc[0]
    
        # Find the time-to-recovery
        tau = t_rec - t_s
        tau_rel = tau/t_end
        
        # Update the metrics dict
        metrics['t_rec'] = t_rec
        metrics['tau'] = tau
        metrics['tau_rel'] = tau_rel
        
        # Find the dropoff avg
        drop_cond = (P['env'] == 1) & (P['episode'] <= t_rec) & (Pbar['episode'] > t_s + N/4)
        
        # Protect against irrecoverable cases (maybe redundant)
        if drop_cond.any():
            Pbar_d = P.loc[drop_cond, 'reward'].mean()
            
            # Find the dropoff delta
            DeltaPbar_d = Pbar_d - Pbar_s
            DeltaPbar_d_rel = DeltaPbar_d/(Pbar_s - Pbar_min) # Relative to switch peak
            DeltaPbar_star_d_rel = DeltaPbar_d/(Pbar_max - Pbar_min) # Relative to max
            
            # Update the metrics dict
            metrics['Pbar_d'] = Pbar_d
            metrics['DeltaPbar_d'] = DeltaPbar_d
            metrics['DeltaPbar_d_rel'] = DeltaPbar_d_rel
            metrics['DeltaPbar*_d_rel'] = DeltaPbar_star_d_rel
    
    return metrics

def performance(dfs: dict, ctrl_vars: dict, N: int) -> dict:
    '''
    Calculate the performance metrics.
    
    Parameters:
    -----------
    dfs: dict
        Raw and moving avg dataframes.
    ctrl_vars: dict
        Control variables (baseline measurements).
    N: int
        Moving avg window.
    
    Returns:
    --------
    dict
        Performance metrics.
    '''
    # Init metrics dict
    metrics = {
        "DeltaPbar": np.nan,
        "DeltaPbar_rel": np.nan,
        "t_XO": np.nan,
        "t_XO_rel": np.nan,
        "T_tot": np.nan,
        "T_tot_rel": np.nan
    }
    
    # Extract the dataframes
    P = dfs['raw']
    Pbar = dfs['Pbar']
    T_lofi = dfs['time']['lofi'] # [s]
    T_hifi = dfs['time']['hifi'] # [s]
    
    # Useful constants
    Pbar_max = Pbar['reward'].max() # Max reward
    Pbar_min_HiFi = ctrl_vars['Pbar_min'][0]
    Pbar_max_HiFi = ctrl_vars['Pbar_max'][0]
    T_max_HiFi = ctrl_vars['T_max'][0]
    t_s = Pbar.loc[Pbar['env'] == 0, 'episode'].iloc[-1] # Switch episode
    t_end = Pbar['episode'].iloc[-1] # End episode
    
    # Std devs
    Pbar_min_HiFi_std = ctrl_vars['Pbar_min'][1]
    Pbar_max_HiFi_std = ctrl_vars['Pbar_max'][1]
    T_max_HiFi_std = ctrl_vars['T_max'][1]
    
    # TODO: Add logic to confidence interval logic
    
    # Find the performance delta
    DeltaPbar = Pbar_max - Pbar_max_HiFi
    DeltaPbar_rel = DeltaPbar/(Pbar_max_HiFi - Pbar_min_HiFi)
    
    # Update the metrics dict
    metrics['DeltaPbar'] = DeltaPbar
    metrics['DeltaPbar_rel'] = DeltaPbar_rel
    
    # Find the crossover episode
    XO_cond = (P['env'] == 1) & (Pbar['reward'].rolling(window=N).min() >= Pbar_max_HiFi)
    
    # Protect against irrecoverable cases
    if XO_cond.any():
        # Find the crossover episode
        t_XO = Pbar.loc[XO_cond, 'episode'].iloc[0]
        t_XO_rel = t_XO/t_end
        
        # Update the metrics dict
        metrics['t_XO'] = t_XO
        metrics['t_XO_rel'] = t_XO_rel
        
        # Estimate the time (in seconds) to crossover
        t_XO_rel_hifi = (t_XO - t_s)/(t_end - t_s)
        T_tot = T_lofi + t_XO_rel_hifi*T_hifi
        T_tot_rel = T_tot/T_max_HiFi
        
        # Update the metrics dict
        metrics['T_tot'] = T_tot
        metrics['T_tot_rel'] = T_tot_rel
    
    return metrics

