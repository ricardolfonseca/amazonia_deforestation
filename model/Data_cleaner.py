# model/Data_cleaner.py

import pandas as pd
from config import *

class Data_cleaner:
    '''
    Class for DataFrame cleaning:
      - remove duplicate rows
      - replace null values with zero
      - create datetime column and set as index
      - filter by time interval
    '''

    def __init__(self, df: pd.DataFrame):
        '''
        Receives a DataFrame (after merge) and stores it internally for cleaning.
        Expects columns 'ano','mes','dia' initially, and that
        the datetime index will be created later.
        '''
        self.__df = df


    def remove_duplicates(self):
        '''
        Removes duplicate rows, keeping only the first occurrence.
        '''
        total_duplicates = self.__df.duplicated().sum()
        if total_duplicates > 0:
            self.__df.drop_duplicates(inplace=True)
            print(f'✅ Duplicates removed: {total_duplicates}')
        else:
            print('✅ No duplicate records found.')


    def replace_nulls_with_zero(self):
        '''
        Replaces all null values in the DataFrame with 0.
        '''
        total_nulls = self.__df.isnull().sum().sum()
        if total_nulls > 0:
            self.__df.fillna(0, inplace=True)
            print(f'✅ Total null values replaced with zero: {total_nulls}')
        else:
            print('✅ No null values to replace.')


    def filter_by_time_interval(self, start_year, start_month, end_year, end_month):
        '''
        Filters the DataFrame between the start and end (year/month), inclusive.
        Ensures there is a datetime index and applies slice by it.
        '''
        start = pd.Timestamp(year=start_year, month=start_month, day=1)
        end   = pd.Timestamp(year=end_year,   month=end_month,   day=1) + pd.offsets.MonthEnd(0)
        self.__df = self.__df.loc[start:end]
        print(f'✅ Filtered from {start.date()} to {end.date()}, total {len(self.__df)} rows.')


    def extend_farmland(self, end_year: int, end_month: int):
        '''
        Ensures the daily index goes up to end_year/end_month, and
        fills 'farmland_area_ha' in new days with the last known value
        in 2023 (or the last day before the cutoff).
        '''
        # 1) Create the full daily index
        start = self.__df.index.min()
        end   = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        full_idx = pd.date_range(start, end, freq='D')
        self.__df = self.__df.reindex(full_idx)

        # 2) Identify the last 'real' farmland value before 2024
        cutoff = pd.Timestamp(year=2023, month=12, day=31)
        hist = self.__df.loc[:cutoff, 'farmland_area_ha'].dropna()
        last_val = hist.iloc[-1] if not hist.empty else 0.0

        # 3) Fill all NaNs (including 2024+) with this value
        self.__df['farmland_area_ha'].fillna(last_val, inplace=True)

        print(f'✅ Farmland extended to {end.date()}, using {last_val:.2f} ha/day for {end_year}-{end_month:02d}.')
        

    def save_and_get_df(self) -> pd.DataFrame:
        '''
        Saves the final DataFrame (with datetime index) to DATASET_CLEAN
        and returns it.
        '''
        self.__df.to_csv(DATASET_CLEAN, index=True, index_label='date')
        print(f'✅ Clean dataset saved to: {DATASET_CLEAN}')
        return self.__df