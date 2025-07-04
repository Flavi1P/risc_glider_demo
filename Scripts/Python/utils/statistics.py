import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import polars as pl

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

def plot_r2_heatmap(r2_df, time_thresh_hr=5, dist_thresh_km=10, r2_min=0.7, r2_max=0.99, r2_step=0.01):
    """
    Plot a heatmap showing the number of unique CTD profiles that have R2 above given thresholds.
    Each cell is annotated with the count.
    """
    # Filter by time and distance thresholds
    df = r2_df[(r2_df['time_diff_hr'] <= time_thresh_hr) & (r2_df['dist_km'] <= dist_thresh_km)]
    
    # Define specific R2 thresholds and variable names as requested
    r2_thresholds = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    variables = ['r2_temp', 'r2_salinity', 'r2_chla', 'r2_bbp', 'r2_doxy']
    var_labels = ['Temp', 'Salinity', 'Chla', 'BBP', 'Doxy']
    
    # Initialize heatmap matrix
    heatmap = np.zeros((len(variables), len(r2_thresholds)), dtype=int)

    # For each variable and threshold, count unique CTD profiles with R2 >= threshold
    for i, var in enumerate(variables):
        for j, thresh in enumerate(r2_thresholds):
            mask = df[var] >= thresh
            unique_ctd = df.loc[mask, 'ctd_profile_id'].nunique()
            heatmap[i, j] = unique_ctd

    # Create the plot
    plt.figure(figsize=(10, 6))
    im = plt.imshow(heatmap, aspect='auto', cmap='PuBu', 
                    extent=[-0.5, len(r2_thresholds)-0.5, -0.5, len(variables)-0.5])
    plt.colorbar(im, label='Number of Unique CTD Profiles')
    
    # Set axis labels and ticks
    plt.yticks(range(len(variables)), var_labels)
    plt.xticks(range(len(r2_thresholds)), [f'{t:.2f}' for t in r2_thresholds])
    plt.xlabel('R² Threshold')
    plt.ylabel('Variable')
    plt.title(f'Unique CTD Profiles with R² ≥ Threshold\n(Time ≤ {time_thresh_hr}hr, Distance ≤ {dist_thresh_km}km)')

    # Annotate each cell with the exact count
    for i in range(len(variables)):
        for j in range(len(r2_thresholds)):
            count = heatmap[i, j]
            # Choose text color based on background intensity
            text_color = 'white' if count > heatmap.max() / 2 else 'black'
            plt.text(j, i, str(count), ha='center', va='center', 
                    color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def get_binned_data(variable_dfs, glider_binned, ctd_binned):
    """
    Link variable_dfs with glider_binned and ctd_binned data to get full binned profile data.
    
    Parameters:
    - variable_dfs: dict, DataFrames for each variable containing paired profiles.
    - glider_binned: polars.DataFrame, binned data for glider profiles.
    - ctd_binned: polars.DataFrame, binned data for CTD profiles.
    
    Returns:
    - dict, updated variable_dfs with linked binned data.
    """
    updated_dfs = {}
    
    for var, df in variable_dfs.items():
        # Convert the current DataFrame to polars for efficient joining
        df_pl = pl.from_pandas(df)
        
        # Join with glider_binned to get binned glider data
        df_with_glider_bins = df_pl.join(
            glider_binned,
            left_on="glider_profile_id",
            right_on="profile_id",
            how="inner"
        )
        
        # Join with ctd_binned to get binned CTD data
        df_with_ctd_bins = df_with_glider_bins.join(
            ctd_binned,
            left_on=["ctd_profile_id", "depth_bin"],
            right_on=["profile_idx", "depth_bin"],
            how="inner"
        )
        
        # Convert back to pandas for compatibility
        updated_dfs[var] = df_with_ctd_bins.to_pandas()
    
    return updated_dfs

def plot_multi_panel_scatter(linked_variable_dfs, var_pairs):
    """
    Create a multi-panel scatter plot comparing CTD and glider values for multiple variables.

    Parameters:
    - linked_variable_dfs: dict, DataFrames for each variable containing paired profiles.
    - var_pairs: list of tuples, each containing (glider_column, ctd_column, variable_name).
    """
    fig, axes = plt.subplots(1, len(var_pairs), figsize=(20, 6), sharex=False, sharey=False)
    
    for ax, (g_col, c_col, var) in zip(axes, var_pairs):
        # Extract data for the current variable
        df = linked_variable_dfs[var]
        x = df[c_col].values
        y = df[g_col].values
        
        # Perform linear regression
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask]
        if len(x_clean) > 0:
            model = LinearRegression()
            model.fit(x_clean, y_clean)
            slope = model.coef_[0]
            intercept = model.intercept_
            equation = f"y = {slope:.2f}x + {intercept:.2f}"
        else:
            equation = "No valid data"

        # Calculate mean distance in space and time
        mean_dist_km = df['dist_km'].mean()
        mean_time_diff_hr = df['time_diff_hr'].mean()

        # Plot scatter
        ax.scatter(x, y, alpha=0.7, edgecolor='k', label=f'{var}')
        
        # Add 1:1 regression line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
        
        # Add regression line
        if len(x_clean) > 0:
            ax.plot(x_clean, model.predict(x_clean), color='blue', linestyle='-', label='Regression Line')
        
        # Set labels and title
        ax.set_xlabel(f'CTD {var}')
        ax.set_ylabel(f'Glider {var}')
        ax.set_title(f'{var}: CTD vs Glider\n{equation}\nMean Dist: {mean_dist_km:.2f} km, Mean Time Diff: {mean_time_diff_hr:.2f} hr')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()