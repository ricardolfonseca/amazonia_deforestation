# controller/controller.py

import pandas as pd
import calendar
from config import *


def merge_datasets():
    """
    Reads the four datasets (deforestation, fires, precipitation, and pasture),
    merges them into a single daily DataFrame, and saves it with the 'date' column as the time index.
    """
    # 1) Load each individual CSV
    df_defor = pd.read_csv(DF_DEFORESTATION)
    df_fires = pd.read_csv(DF_FIRES)
    df_prec  = pd.read_csv(DF_PRECIPITATION)
    df_farm  = pd.read_csv(DF_PASTURE)

    # 2) Extract year, month, day from precipitation (if needed)
    df_prec['date'] = pd.to_datetime(df_prec['date'])
    df_prec['year'] = df_prec['date'].dt.year
    df_prec['month'] = df_prec['date'].dt.month
    df_prec['day'] = df_prec['date'].dt.day
    df_prec = df_prec.rename(columns={'precipitation': 'precipitation_mm'})
    df_prec = df_prec[['year', 'month', 'day', 'precipitation_mm']]

    # 3) Rename other columns to avoid collisions and use English names
    df_defor = df_defor.rename(columns={'area_ha': 'deforestation_area_ha'})
    df_fires = df_fires.rename(columns={'focos': 'fires'})
    df_farm  = df_farm.rename(columns={'area_ha': 'pasture_area_ha'})

    # 4) Merge daily data: deforestation + fires + precipitation
    df = (
        df_defor
        .merge(df_fires, on=['year', 'month', 'day'], how='outer')
        .merge(df_prec,  on=['year', 'month', 'day'], how='outer')
    )

    # 5) pasture annual → daily average
    df_farm['days_in_year'] = df_farm['year'].apply(
        lambda y: 366 if calendar.isleap(y) else 365
    )
    df_farm['pasture_area_ha'] = (
        df_farm['pasture_area_ha'] / df_farm['days_in_year']
    )

    # Keep only what is needed (year + daily area) – remove days_in_year
    df_farm = df_farm[['year', 'pasture_area_ha']]

    # 6) Merge by 'year' – replicate daily
    df = df.merge(df_farm, on='year', how='left')

    # 7) Create 'date' column and remove year, month, day
    required_cols = {'year', 'month', 'day'}
    if required_cols.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.drop(columns=['year', 'month', 'day'], inplace=True)
    else:
        raise ValueError(f"❌ Columns {required_cols} are required to generate the date.")

    # 8) Sort and save the final CSV
    df = df.sort_values('date').set_index('date')
    df.to_csv(DATASET_MERGED, index=True)
    print(f"✅ Complete dataset saved at: {DATASET_MERGED}")

    return df
