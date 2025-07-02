import numpy as np
import pandas as pd

def moving_avg(df: pd.DataFrame, col: str, window: int=50, *args, **kwargs) -> pd.DataFrame:
    """
    Moving average filter.
    
    Parameters:
    -----------
    df: pd.DataFrame
        Data to filter.
    col: str
        Column name.
    window: int
        Window size for the moving average.
        
    Returns:
    --------
    pd.DataFrame
        Filtered data.
    """
    df_f = df.copy()
    df_f[col] = df[col].rolling(window=window).mean()
    return df_f.dropna()
    

def kalman(df: pd.DataFrame, col: str, proc_var: float=1e-5, mes_var: float=1, *args, **kwargs) -> pd.DataFrame:
    """
    Apply Kalman filter to the data.
    
    Parameters:
    -----------
    df: pd.DataFrame
        Data to filter.
    col: str
        Column name.
    proc_var: float
        Process variance.
    mes_var: float
        Measurement variance.
    
    Returns:
    --------
    pd.DataFrame
        Filtered data.
    """
    df_f = df.copy()
    n = len(df_f)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)
    
    xhat[0] = df_f[col][0]
    P[0] = proc_var
    
    for k in range(1, n):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + proc_var
        
        K[k] = Pminus[k] / (Pminus[k] + mes_var)
        xhat[k] = xhatminus[k] + K[k] * (df_f[col][k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    df_f[col] = xhat
    return df_f