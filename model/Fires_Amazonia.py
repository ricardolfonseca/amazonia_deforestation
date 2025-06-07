# model/Fires_amazonia.py

import os
import geopandas as gpd
import pandas as pd
import ee
from shapely.geometry import mapping

from config import *

class Fires_amazonia:
    '''
    Processes fire hotspots in the Legal Amazonia for the specified period,
    generating a CSV with the daily count of FIRMS alerts.
    '''

    def __init__(self):
        # year range and directories
        self.__start_year = START_YEAR
        self.__end_year = END_YEAR
        self.__data_dir = FIRES_DIR

        # Raw and output file paths
        self.__raw_path = os.path.join(
            self.__data_dir,
            f'fires_{self.__start_year}_{self.__end_year}.geojson'
        )
        self.__csv_path = DF_FIRES
        
        # Path to the Earth Engine credentials file
        self.__credentials = EE_CREDENTIALS


    def init_ee(self):
        '''Initializes the Earth Engine API if not already initialized.'''
        if not ee.data._initialized:
            ee.Initialize(ee.ServiceAccountCredentials(None, self.__credentials))


    def download_data(self):
        '''
        Exports FIRMS fire hotspot points via Earth Engine to GeoJSON.
        Saves to self.__raw_path; does nothing if the file already exists.
        '''
        if os.path.exists(self.__raw_path):
            return
        self.init_ee()

        # 1) Create an ee.Geometry of the Legal Amazonia from the local shapefile
        lim_gdf = gpd.read_file(SHAPEFILE_AMAZONIA).to_crs('EPSG:4326')
        coords = mapping(lim_gdf.unary_union)['coordinates']
        region = ee.Geometry.Polygon(coords)

        # 2) Filter FIRMS by date and within the Legal Amazonia
        start = f'{self.__start_year}-01-01'
        end = f'{self.__end_year}-12-31'
        fires = (
            ee.ImageCollection('FIRMS')
              .filterDate(start, end)
              .filterBounds(region)
        )

        def extract(img):
            '''
            For each FIRMS image:
            -- creates a mask of hot pixels (T21 > 330 K)
            -- vectorizes using the mask geometry within the region
            -- attaches the image date in 'acq_date'
            '''
            mask = img.select('T21').gt(330).selfMask()
            pts = mask.reduceToVectors(
                geometry=region,
                geometryType='centroid',
                scale=1000,
                reducer=ee.Reducer.countEvery(),
                maxPixels=1e13
            )
            return pts.map(lambda f: f.set(
                'acq_date', img.date().format('YYYY-MM-dd')
            ))

        points = fires.map(extract).flatten()

        task = ee.batch.Export.table.toCloudStorage(
            collection=points,
            description='export_fires',
            bucket='amazonia-fires',
            fileNamePrefix=f'fires_{self.__start_year}_{self.__end_year}',
            fileFormat='GeoJSON'
        )
        task.start()
        raise RuntimeError(
            f'Wait for export and move the GeoJSON to: {self.__raw_path}'
        )


    def process_shapefile(self):
        # 1) Load the raw GeoJSON file
        if not os.path.exists(self.__raw_path):
            self.download_data()                    # Download data if not present
        gdf = gpd.read_file(self.__raw_path)

        # 2) Load Legal Amazonia shapefile and reproject
        lim = gpd.read_file(SHAPEFILE_AMAZONIA).to_crs(gdf.crs)
        union = lim.unary_union

        # 3) Filter only points within the Legal Amazonia
        gdf = gdf[gdf.geometry.within(union)]

        # 4) Convert date and aggregate by day
        if 'acq_date' not in gdf.columns:
            raise ValueError(f"Column 'acq_date' not found.")
        gdf['date'] = pd.to_datetime(gdf['acq_date'])

        df = (
            gdf
            .groupby(gdf['date'].dt.date)
            .size()
            .reset_index(name='fires')
        )
        df.rename(columns={'index': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        # 5) Extract year, month, day and save
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df[['year', 'month', 'day', 'fires']].to_csv(self.__csv_path, index=False)
        print(0, f'✅ Fire hotspots saved to: {self.__csv_path}')

        return df[['year', 'month', 'day', 'fires']]
