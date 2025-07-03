import numpy as np
def type2_regression_r2(x, y):
    """
    Compute R2 for type 2 regression (major axis regression)

    input:
    - x: numpy array of independent variable values
    - y: numpy array of dependent variable values
    output:
    - r2: R-squared value for the regression
    """
    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan
    
    # Type 2 regression slope
    s_yx = np.std(y_clean, ddof=1) / np.std(x_clean, ddof=1)
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    slope = np.sign(r) * s_yx
    intercept = np.mean(y_clean) - slope * np.mean(x_clean)
    
    # Create a simple linear model for pybroom compatibility
    # We'll use the type 2 regression parameters we calculated
    y_pred = slope * x_clean + intercept
    
    # Calculate R2
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    return r2