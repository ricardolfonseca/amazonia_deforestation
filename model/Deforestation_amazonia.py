# Deforestation_amazonia.py

import os
import geopandas as gpd
import pandas as pd

from config import *

class Deforestation_amazonia:
    '''
    Processes DETER data for Legal Amazonia, transforming the accumulated mask
    into a daily deforestation report in hectares.
    '''

    def __init__(self):
        self.__alerts_path = SHAPEFILE_ALERTS           # Path to the shapefile with DETER alerts
        self.__csv_path = DF_DEFORESTATION              # Directory and path for the output CSV


    def process_shapefile(self):
        # 1) Read the shapefile
        gdf = gpd.read_file(self.__alerts_path)

        # 2) Ensure proper projection for area calculation
        if gdf.crs is None or not gdf.crs.is_projected:
            gdf = gdf.to_crs('EPSG:5880')

        # 3.1) Identify date column and convert
        for col in ('IMAGE_DATE', 'VIEW_DATE', 'data_alerta'):
            if col in gdf.columns:
                gdf['date'] = pd.to_datetime(gdf[col], errors='coerce')
                break
        else:
            raise ValueError('Date column not found in shapefile.')

        # 3.2)  Drop lines with missing values in date
        gdf = gdf.dropna(subset=['date'])

        # 4.1) Fix invalid geometries
        gdf['geometry'] = gdf['geometry'].buffer(0)
        gdf = gdf[gdf.is_valid]

        # 4.2) Dissolve by day to merge polygons and remove overlaps
        daily_union = (
            gdf
            .dissolve(by='date')
            .reset_index()[['date', 'geometry']]
            .assign(accum_area_ha=lambda df: df.geometry.area / 10_000)
        )

        # If 'area_desflorestacao_ha' exists, rename to 'deforestation_area_ha'
        # if 'area_desflorestacao_ha' in daily_union.columns:
        #    daily_union = daily_union.rename(columns={'area_desflorestacao_ha': 'deforestation_area_ha'})

        # 5) Extract year, month, and day
        daily_union['year'] = daily_union['date'].dt.year
        daily_union['month'] = daily_union['date'].dt.month
        daily_union['day'] = daily_union['date'].dt.day

        # 6) Sort and calculate daily delta (new hectares per day)
        daily_union = daily_union.sort_values('date')
        daily_union['deforestation_area_ha'] = (
            daily_union['accum_area_ha']
            .diff()
            .fillna(daily_union['accum_area_ha'])
            .clip(lower=0)
        )

        # 7) Prepare final DataFrame and save
        df = daily_union[['year', 'month', 'day', 'deforestation_area_ha']].copy()
        df.to_csv(self.__csv_path, index=False)
        print(f'✅ Daily deforestation saved at: {self.__csv_path}')

        return df