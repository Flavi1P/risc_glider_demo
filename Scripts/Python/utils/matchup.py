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
