def read_ctd(filepath):
    """This function reads a CTD data file, extracts metadata, and returns a DataFrame with the data.

    Args:
        filepath (string): The full path of the ctd file to read.

    Returns:
        pandas dataframe : A pandas DataFrame containing the CTD data with metadata included.
    """    
    import pandas as pd
    import re
    from datetime import datetime
    # Read column names from line 27 (R skips 26 lines, so read line 27 as header)
    cols = pd.read_csv(filepath, skiprows=26, nrows=0)
    cols_name = cols.columns.tolist()

    # Read actual data (R skips 28 lines, so data starts at line 29)
    dat = pd.read_csv(filepath, skiprows=28, header=None)
    dat.columns = cols_name

    # Read raw lines for metadata
    with open(filepath, "r") as f:
        raw_dat = f.readlines()

    # Extract LON
    lon_line = next(line for line in raw_dat if "LON" in line)
    lon = float(re.search(r"=(.*)", lon_line).group(1).strip())

    # Extract LAT
    lat_line = next(line for line in raw_dat if "LAT" in line)
    lat = float(re.search(r"=(.*)", lat_line).group(1).strip())

    # Add to dataframe
    dat["lon"] = lon
    dat["lat"] = lat

    # Extract DATE (second match, as in R)
    date_lines = [line for line in raw_dat if "DATE" in line]
    date_str = re.search(r"=(.*)", date_lines[1]).group(1).strip()
    date_val = pd.to_datetime(date_str).date()

    # Extract TIME
    time_line = next(line for line in raw_dat if "TIME" in line)
    time_str = re.search(r"= (.*)", time_line).group(1).strip()

    # Parse TIME (e.g. "1025" → "10:25")
    time_str_formatted = re.sub(r"^([0-9]{2})([0-9]+)$", r"\1:\2", time_str)
    time_val = pd.to_datetime(time_str_formatted, format="%H:%M").time()

    # Combine date and time
    datetime_val = datetime.combine(date_val, time_val)

    # Add datetime to dataframe
    dat["datetime"] = datetime_val
    return dat


def read_glider_og1(filepath, varlist = ["TIME", "DEPTH", "TEMP", "CHLA", "BBP700", "ABS_SALINITY", "MOLAR_DOXY", "PROFILE_NUMBER", "LATITUDE", "LONGITUDE"], to_polars=True):
    """Open a glider OG1 file, extract glider name, and return a DataFrame with the data. Can be either a pandas DataFrame or a polars DataFrame.

    Args:
        filepath (string): The path of the glider file to read.
        varlist (list, optional): list. Defaults to ["TIME", "DEPTH", "TEMP", "CHLA", "BBP700", "ABS_SALINITY", "MOLAR_DOXY", "PROFILE_NUMBER", "LATITUDE", "LONGITUDE"].
        to_polars (bool, optional): list. Defaults to True.

    Returns:
        dataframe: A dataframe with the selected variables, either polars df or pandas df.
    """    
    import xarray as xr
    import pandas as pd
    ds = xr.open_dataset(filepath, decode_times=True)
    glider_name = ds.attrs.get("trajectory").split("_")[0]
    # Select variables of interest
    ds_sel = ds[varlist]
    # Convert to pandas DataFrame, then to polars DataFrame
    df_pd = ds_sel.to_dataframe().reset_index()

    if to_polars:
        import polars as pl
        df_pl = pl.from_pandas(df_pd)
        # Add glider name
        df_pl = df_pl.with_columns(pl.lit(glider_name).alias("glider_name"))
        return(df_pl)
    else:
        # Add glider name
        df_pd["glider_name"] = glider_name
        return df_pd