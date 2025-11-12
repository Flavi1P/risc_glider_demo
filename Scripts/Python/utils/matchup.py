def find_candidate_glider_ctd_pairs(glider_df, ctd_df, time_thresh_hr=2, dist_thresh_km=5):
    """
    Identify all pairs of glider and CTD profiles within specified time (hours) and distance (km) thresholds.
    
    Returns a DataFrame of matching pairs with details.
    """
    #TODO: "Add checker for datetime format"
    matches = []
    
    for _, g_row in glider_df.iterrows():
        for _, c_row in ctd_df.iterrows():
            # Time difference
            time_diff_hr = abs((g_row['median_datetime'] - c_row['datetime']).total_seconds()) / 3600.0
            if time_diff_hr > time_thresh_hr:
                continue

            # Skip if coordinates are missing
            if pd.isnull(g_row['median_latitude']) or pd.isnull(g_row['median_longitude']) \
               or pd.isnull(c_row['lat']) or pd.isnull(c_row['lon']):
                continue

            # Distance in kilometers
            dist_km = geodesic(
                (g_row['median_latitude'], g_row['median_longitude']),
                (c_row['lat'], c_row['lon'])
            ).km

            if dist_km > dist_thresh_km:
                continue

            matches.append({
                'glider_profile_id': g_row['profile_id'],
                'glider_name': g_row['glider_name'],
                'ctd_profile_id': c_row.get('profile_idx', None),  # default to None if not present
                'time_diff_hr': time_diff_hr,
                'dist_km': dist_km
            })
    
    return pd.DataFrame(matches)

def find_candidate_glider_pairs(glider_a_df, glider_b_df, time_thresh_hr=2, dist_thresh_km=5):
    """
    Identify all pairs of glider and CTD profiles within specified time (hours) and distance (km) thresholds.
    
    Returns a DataFrame of matching pairs with details.
    """
    matches = []
    
    for _, g_row in glider_a_df.iterrows():
        for _, c_row in glider_b_df.iterrows():
            # Time difference
            time_diff_hr = abs((g_row['median_datetime'] - c_row['median_datetime']).total_seconds()) / 3600.0
            if time_diff_hr > time_thresh_hr:
                continue

            # Skip if coordinates are missing
            if pd.isnull(g_row['median_latitude']) or pd.isnull(g_row['median_longitude']) \
               or pd.isnull(c_row['median_latitude']) or pd.isnull(c_row['median_longitude']):
                continue

            # Distance in kilometers
            dist_km = geodesic(
                (g_row['median_latitude'], g_row['median_longitude']),
                (c_row['median_latitude'], c_row['median_longitude'])
            ).km

            if dist_km > dist_thresh_km:
                continue

            matches.append({
                'glider_a_profile_id': g_row['profile_id'],
                'glider_name': glider_a,
                'glider_b_profile_id': c_row.get('profile_id', None),  # default to None if not present
                'glider_b_name': glider_b,
                'time_diff_hr': time_diff_hr,
                'dist_km': dist_km
            })
    
    return pd.DataFrame(matches)

import pandas as pd
from geopy.distance import geodesic

def find_candidate_profile_pairs(df1, df2,
                                  datetime_col1, lat_col1, lon_col1,
                                  datetime_col2, lat_col2, lon_col2,
                                  id_col1, id_col2,
                                  name_col1=None, name_col2=None,
                                  time_thresh_hr=2, dist_thresh_km=5,
                                  label1='profile1', label2='profile2'):
    """
    General function to identify matching profile pairs between two datasets 
    within specified time (hours) and distance (km) thresholds.

    Parameters:
    - df1, df2: DataFrames containing the profile data.
    - datetime_col1/2: Column names for datetime.
    - lat_col1/2, lon_col1/2: Column names for latitude and longitude.
    - id_col1/2: Column names for profile IDs.
    - name_col1/2: Optional column names for glider names.
    - label1/2: Labels used to prefix the keys in the output dictionary.
    - time_thresh_hr, dist_thresh_km: Matching thresholds.

    Returns:
    - DataFrame of matching pairs with time and distance differences.
    """
    matches = []

    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            # Time difference
            try:
                time_diff_hr = abs((row1[datetime_col1] - row2[datetime_col2]).total_seconds()) / 3600.0
            except Exception:
                continue  # Skip if datetime is invalid
            if time_diff_hr > time_thresh_hr:
                continue

            # Skip if coordinates are missing
            if pd.isnull(row1[lat_col1]) or pd.isnull(row1[lon_col1]) \
               or pd.isnull(row2[lat_col2]) or pd.isnull(row2[lon_col2]):
                continue

            # Distance in kilometers
            dist_km = geodesic(
                (row1[lat_col1], row1[lon_col1]),
                (row2[lat_col2], row2[lon_col2])
            ).km

            if dist_km > dist_thresh_km:
                continue

            match = {
                f'{label1}_profile_id': row1[id_col1],
                f'{label2}_profile_id': row2.get(id_col2, None),
                'time_diff_hr': time_diff_hr,
                'dist_km': dist_km
            }

            if name_col1:
                match[f'{label1}_name'] = row1.get(name_col1, None)
            if name_col2:
                match[f'{label2}_name'] = row2.get(name_col2, None)

            matches.append(match)

    return pd.DataFrame(matches)

def interpolate_over_time(ds, var_name, method='linear'):
    import numpy as np
    import xarray as xr
    from scipy import interpolate
    """
    Interpolate missing values for any specified variable over a given dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to interpolate.
    var_name : str
        Name of the variable to interpolate (e.g., 'DEPTH', 'TEMP', 'BBP700').
    dim_name : str, optional
        Name of the dimension or coordinate to interpolate along (default: 'TIME').
    method : str, optional
        Interpolation method: 'linear', 'nearest', etc. (default: 'linear').

    Returns
    -------
    ds : xarray.Dataset
        Dataset with an added variable named <var_name>_INTERP.
    """

    if var_name not in ds.variables:
        raise ValueError(f"Variable '{var_name}' not found in dataset")



    var = ds[var_name].values.copy()
    coord = ds["TIME"].values

    # Convert datetime coordinates to numeric (seconds since start)
    if np.issubdtype(coord.dtype, np.datetime64):
        coord_seconds = (coord - coord[0]) / np.timedelta64(1, "s")
    else:
        coord_seconds = coord

    n_total = len(var)
    n_valid = np.sum(np.isfinite(var))
    n_missing = n_total - n_valid

    print(f"\nInterpolating variable: {var_name}")
    print(f"  Total points: {n_total}")
    print(f"  Valid: {n_valid} ({100 * n_valid / n_total:.1f}%)")
    print(f"  Missing: {n_missing} ({100 * n_missing / n_total:.1f}%)")

    if n_missing == 0:
        print("  No interpolation needed — all values are valid.")
        ds[f"{var_name}_INTERP"] = ds[var_name].copy()
        return ds

    if n_valid < 2:
        print("  ERROR: Need at least 2 valid points for interpolation.")
        return ds

    valid_mask = np.isfinite(var) & np.isfinite(coord_seconds)
    if np.sum(valid_mask) < 2:
        print("  ERROR: Need at least 2 valid coordinate-value pairs.")
        return ds

    # Build interpolator
    f_interp = interpolate.interp1d(
        coord_seconds[valid_mask],
        var[valid_mask],
        kind=method,
        bounds_error=False,
        fill_value="extrapolate",
    )

    var_interp = f_interp(coord_seconds)

    # Prevent extrapolation beyond valid range
    first_valid = np.where(valid_mask)[0][0]
    last_valid = np.where(valid_mask)[0][-1]
    var_interp[:first_valid] = np.nan
    var_interp[last_valid + 1 :] = np.nan

    # Add new variable to dataset
    ds[f"{var_name}_INTERP"] = xr.DataArray(
        var_interp,
        dims=ds[var_name].dims,
        coords=ds[var_name].coords,
        attrs=ds[var_name].attrs.copy(),
    )
    ds[f"{var_name}_INTERP"].attrs["comment"] = (
        f"Interpolated {var_name} along TIME (method={method})"
    )

    n_filled = np.sum(np.isfinite(var_interp)) - n_valid
    print(f"  Filled {n_filled} missing values by interpolation.")

    return ds