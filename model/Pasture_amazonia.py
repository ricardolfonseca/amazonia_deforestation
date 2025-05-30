# model/pasture_amazonia.py

import os
import ee
import pandas as pd
import geopandas as gpd

from config import *

class Pasture_amazonia:
    '''
    Extracts the annual pasture area in Legal Amazon
    (class 15 in MapBiomas Collection 9) and generates a CSV
    with one row per year: year, pasture_area_ha.
    '''

    def __init__(self):
        self.__start_year = START_YEAR
        self.__end_year   = 2023                        # last complete year
        self.__data_dir   = PASTURE_DIR
        self.__csv_path  = DF_PASTURE
        self.__credentials    = EE_CREDENTIALS          # Earth Engine credentials


    def init_ee(self):
        '''Initializes the Earth Engine API if necessary.'''
        if not ee.data._initialized:
            ee.Initialize(ee.ServiceAccountCredentials(None, self.__credentials))


    def process(self):
        '''
        1) Initializes EE and loads the Legal Amazon boundary
        2) For each year in the interval, selects the classification band
           from MapBiomas, filters the 'pasture' class (code 15),
           sums the area (m² → ha) over the entire Legal Amazon
        3) Builds a DataFrame with columns: year, pasture_area_ha
        4) Saves to CSV and returns the DataFrame
        '''
        self.init_ee()
        amz = gpd.read_file(SHAPEFILE_AMAZONIA).to_crs('EPSG:4326')         # read shapefile of Legal Amazon and reproject to lon/lat
        region = ee.Geometry(amz.unary_union.__geo_interface__)

        asset = (                                                           # MapBiomas Collection 9 asset
            'projects/mapbiomas-public/assets/'     
            'brazil/lulc/collection9/'
            'mapbiomas_collection90_integration_v1'
        )

        records = []

        for year in range(self.__start_year, self.__end_year + 1):          # anual loop

            img = ee.Image(asset).select(f'classification_{year}')          # classification image for the year
            mask = img.eq(15)                                               # mask for the 'pasture' class (code 15)   
            area_m2 = mask.multiply(ee.Image.pixelArea()).reduceRegion(     # sum pixel area (m²)
                reducer   = ee.Reducer.sum(),
                geometry  = region,
                scale     = 30,
                maxPixels = 1e13
            ).get(f'classification_{year}')

            pasture_area_ha = ee.Number(area_m2).divide(10000).getInfo()   # convert to 'ha' and get value
            records.append({'year': year, 'pasture_area_ha': pasture_area_ha})

        df = pd.DataFrame(records)
        df.to_csv(self.__csv_path, index=False)
        print(f'✅ Pasture processed and saved to: {self.__csv_path}')     # Save the DataFrame to CSV
        return df