import numpy as np

def kalman(data, process_variance=1e-5, measurement_variance=1):
    """
    Basic Kalman filter implementation.
    
    Parameters:
    -----------
    data: np.array
        Data to filter.
    process_variance: float
        Process variance.
    measurement_variance: float
        Measurement variance.
        
    Returns:
    --------
    np.array
        Filtered data.
    """
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