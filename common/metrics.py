import pandas as pd
import numpy as np

def _assign_env(df: pd.DataFrame, pct: float):
    """
    Add an 'env' column: 0=LoFi, 1=HiFi, and return switch episode t_s.
    
    For pct==100.0 (pure LoFi): env=0 everywhere, t_s = last episode.
    For pct==0.0   (pure HiFi): env=1 everywhere, t_s = -1.
    Otherwise detect the big jump in episode numbering.
    """
    df = df.copy()
    if pct == 100.0:
        # all LoFi
        df['env'] = 0
        t_s = int(df['episode'].iloc[-1])
    elif pct == 0.0:
        # all HiFi
        df['env'] = 1
        t_s = -1
    else:
        # mixed run: look for the first jump >1
        diffs = df['episode'].diff()
        jumps = diffs[diffs > 1].index
        if len(jumps) > 0:
            switch_idx = jumps[0]
            t_s = int(df.loc[switch_idx - 1, 'episode'])
            df['env'] = (df['episode'] > t_s).astype(int)
        else:
            # fallback: treat as all LoFi
            df['env'] = 0
            t_s = int(df['episode'].iloc[-1])
    return df, t_s

def dropoff(aggregated: dict, N: int) -> dict:
    """
    Compute dropoff metrics for every algo/pct in `aggregated`.
    Skips pure runs (pct 0.0 or 100.0).
    Returns nested dict[algo][pct]→metrics.
    """
    all_m = {}
    for algo, pct_map in aggregated.items():
        all_m[algo] = {}
        for pct, df in pct_map.items():
            if pct in (0.0, 100.0):
                continue

            # prepare
            Pbar = df[['episode', 'mean_reward']].rename(columns={'mean_reward': 'reward'})
            Pbar, t_s = _assign_env(Pbar, pct)
            t_end = int(Pbar['episode'].max())
            Pmin, Pmax = Pbar['reward'].min(), Pbar['reward'].max()
            Pbar_s = float(Pbar.loc[Pbar['env']==0, 'reward'].tail(N+1).mean())

            m = dict(
                t_rec=np.nan, tau=np.nan, tau_rel=np.nan,
                Pbar_s=Pbar_s,
                Pbar_d=np.nan, DeltaPbar_d=np.nan,
                DeltaPbar_d_rel=np.nan, DeltaPbar_star_d_rel=np.nan
            )

            # recovery
            rec = ((Pbar['env']==1) &
                   (Pbar['reward'] >= Pbar_s) &
                   (Pbar['episode'] > t_s + N/4))
            if rec.any():
                t_rec = int(Pbar.loc[rec, 'episode'].iloc[0])
                tau = t_rec - t_s
                m.update(t_rec=t_rec, tau=tau, tau_rel=tau/t_end)

                drop = ((Pbar['env']==1) &
                        (Pbar['episode'] <= t_rec) &
                        (Pbar['episode'] > t_s + N/4))
                if drop.any():
                    Pbar_d = float(Pbar.loc[drop, 'reward'].mean())
                    delta = Pbar_d - Pbar_s
                    m.update(
                        Pbar_d=Pbar_d,
                        DeltaPbar_d=delta,
                        DeltaPbar_d_rel=delta/(Pbar_s - Pmin) if Pbar_s!=Pmin else np.nan,
                        DeltaPbar_star_d_rel=delta/(Pmax - Pmin) if Pmax!=Pmin else np.nan
                    )

            all_m[algo][pct] = m
    return all_m

def performance(aggregated: dict,
                avg_times: dict,
                N: int) -> dict:
    """
    Compute performance metrics vs. the 100%‐LoFi baseline.
    Needs avg_times[algo][pct]['lofi_duration'] & ['hifi_duration'].
    Returns nested dict[algo][pct]→metrics.
    """
    perf = {}
    for algo, pct_map in aggregated.items():
        if 100.0 not in pct_map or algo not in avg_times:
            continue

        # baseline is 100% LoFi → switch to HiFi never happens
        base_df = pct_map[100.0][['episode','mean_reward']].rename(columns={'mean_reward':'reward'})
        base_Pbar, _ = _assign_env(base_df, 100.0)
        Pmin, Pmax = base_Pbar['reward'].min(), base_Pbar['reward'].max()
        T_max = avg_times[algo][100.0]['lofi_duration']  # pure LoFi time

        perf[algo] = {}
        for pct, df in pct_map.items():
            Pbar_df = df[['episode','mean_reward']].rename(columns={'mean_reward':'reward'})
            Pbar, t_s = _assign_env(Pbar_df, pct)
            t_end = int(Pbar['episode'].max())

            deltaP = float(Pbar['reward'].max() - Pmax)
            deltaP_rel = deltaP/(Pmax - Pmin) if Pmax!=Pmin else np.nan

            m = dict(
                DeltaPbar=deltaP,
                DeltaPbar_rel=deltaP_rel,
                t_XO=np.nan, t_XO_rel=np.nan,
                T_tot=np.nan, T_tot_rel=np.nan
            )

            # only mixed runs can crossover
            if pct not in (0.0, 100.0):
                roll_min = Pbar['reward'].rolling(window=N).min()
                xo = (Pbar['env']==1) & (roll_min >= Pmax)
                if xo.any():
                    t_xo = int(Pbar.loc[xo, 'episode'].iloc[0])
                    t_xo_rel = t_xo/t_end
                    frac_hi = (t_xo - t_s)/(t_end - t_s) if (t_end>t_s) else np.nan

                    T_lo = avg_times[algo][pct]['lofi_duration']
                    T_hi = avg_times[algo][pct]['hifi_duration']
                    Ttot = T_lo + frac_hi * T_hi
                    m.update(
                        t_XO=t_xo, t_XO_rel=t_xo_rel,
                        T_tot=Ttot,
                        T_tot_rel=Ttot/T_max if T_max>0 else np.nan
                    )

            perf[algo][pct] = m
    return perf

def performance_contour(perf_metrics: dict) -> dict:
    """
    Convert perf_metrics to DataFrames keyed by algo.
    """
    return {algo: pd.DataFrame.from_dict(pm, orient='index')
            for algo, pm in perf_metrics.items()}

def transfer_gap(perf_metrics: dict) -> dict:
    """
    Extract only DeltaPbar for each algo/pct.
    """
    return {algo: {pct: v['DeltaPbar'] 
                   for pct, v in pm.items()}
            for algo, pm in perf_metrics.items()}
