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

