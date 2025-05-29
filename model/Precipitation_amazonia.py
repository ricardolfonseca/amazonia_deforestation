# model/Precipitation_amazonia.py

import os
import datetime
import pandas as pd 
import geopandas as gpd
import ee

from config import *

class Precipitation_amazonia:
    '''
    Exports, month by month, the average daily precipitation over the entire
    Legal Amazon using CHIRPS (UCSB-CHG/CHIRPS/DAILY)
    via Earth Engine, generating a CSV per month with columns:
    date (YYYY-MM-DD) and precipitation (mm).
    '''

    def __init__(self):
        self.__start_year = START_YEAR
        self.__end_year   = END_YEAR
        self.__data_dir   = PRECIPITATION_DIR
        self.__credentials      = EE_CREDENTIALS                     # Path to the Earth Engine credentials file
        self.__gcbucket        = BUCKET_PRECIPITATION                # Google Cloud Storage bucket for precipitation data
        self.__monthly_template = 'rain_amazon_{year}_{month:02d}.csv'
        self.__daily_template = 'precipitation_amazon_daily.csv'


    def init_ee(self):
        '''Initializes the Earth Engine API if not already initialized.'''
        if not ee.data._initialized:
            ee.Initialize(ee.ServiceAccountCredentials(None, self.__credentials))


    def fetch_and_export_month(self, year, month):
        '''Exports a monthly CSV to the bucket, returning the task.'''
        filename = self.__monthly_template.format(year=year, month=month)
        local_path = os.path.join(self.__data_dir, filename)
        if os.path.exists(local_path):
            print(f'Precipitation CSV already exists locally: {filename}')
            return None

        self.init_ee()
        amz = gpd.read_file(SHAPEFILE_AMAZONIA).to_crs('EPSG:4326')                 # load Legal Amazonia boundary
        region = ee.Geometry(amz.unary_union.__geo_interface__)

        start = datetime.date(year, month, 1)                                       # month period
        end = (start.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)   # next month, then set to the first day of that month
        iso_start, iso_end = start.isoformat(), end.isoformat()                     # convert to ISO format


        chirps = (
            ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')                             # CHIRPS daily precipitation
            .filterDate(iso_start, iso_end)
            .select('precipitation')
        )

        def make_feature(img):                                                      # map each image to a Feature(date, mean_precip)
            date = img.date().format('YYYY-MM-dd')
            mean = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=5000,
                maxPixels=1e13
            ).get('precipitation')
            return ee.Feature(None, {'date': date, 'precipitation': mean})

        fc = chirps.map(make_feature)


        task = ee.batch.Export.table.toCloudStorage(                                # export to Cloud Storage
            collection     = fc,
            description    = filename,
            bucket         = self.__gcbucket,
            fileNamePrefix = filename.replace('.csv',''),
            fileFormat     = 'CSV'
        )
        task.start()
        print(f'Precipitation Task started: {filename} (ID: {task.id})')
        return task


    def prepare_dataset(self):
        '''
        Launches monthly tasks for each month in the period or does final merge if already available locally.
        '''
        merged_path = os.path.join(self.__data_dir, self.__daily_template)
        if os.path.exists(merged_path):
            print(f'[Precipitation] Final merged file created: {merged_path}')
            return

        tasks = []
        for year in range(self.__start_year, self.__end_year + 1):
            for month in range(1, 13):
                t = self.fetch_and_export_month(year, month)
                if t:
                    tasks.append((year, month, t.id))

        if tasks:
            print('\nTasks launched (wait for completion in GCS):')
            for year, month, tid in tasks:
                print(f'  {year}-{month:02d}: {tid}')
        else:
            # Generate list of expected paths (only checks physical existence, not content)
            expected_files = [
                os.path.join(self.__data_dir, self.__monthly_template.format(year=year, month=month))
                for year in range(self.__start_year, self.__end_year + 1)
                for month in range(1, 13)
            ]

            # Continue if ALL files exist, even if empty
            if all(os.path.exists(f) for f in expected_files):
                dfs = []
                for f in expected_files:
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            dfs.append(df)
                        else:
                            print(f'Precipitation Ignored (empty): {f}')
                    except pd.errors.EmptyDataError:
                        print(f'Precipitation Ignored (empty CSV error): {f}')

                if dfs:
                    df_merged = pd.concat(dfs)
                    df_merged.to_csv(merged_path, index=False)
                    print(f'[Precipitation] Final merged file created: {merged_path}')
                else:
                    print('[Precipitation] No valid file to merge.')
            else:
                print('[Precipitation] Not all monthly CSVs are available on disk.')