# model/Eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *

class Eda:
    '''
    Class responsible for performing Exploratory Data Analysis (EDA)
    on the cleaned Amazonia Legal dataset.
    Generates visualizations and saves them in the output folder.
    '''

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)                          # Confirms directory structure and loads the dataset
        self.__df = pd.read_csv(
            DATASET_CLEAN,
            index_col='date'
        )


    def numeric_statistics(self):
        '''
        Calcula e exibe as estatísticas descritivas (count, mean, std, min, 25%, 50%, 75%, max)
        para todas as colunas numéricas, e salva em CSV.
        '''
        num = self.__df.select_dtypes(include='number')                 # Selects only numeric columns
        stats = num.describe().transpose()                              # Transpose for better readability     

        stats[['mean','std','min','25%','50%','75%','max']] = (         # Rounds the statistics to 2 decimal places
            stats[['mean','std','min','25%','50%','75%','max']].round(2)
        )

        print('\n🔢 Numeric statistics :\n', stats)
        path = os.path.join(OUTPUT_DIR, 'numeric_statistics.csv')       # Saves the statistics to a CSV file
        stats.to_csv(path)
        print(f'✅ Statistics saved in: {path}')


    def correlation_matrix(self):
        '''
        Generates and saves a heatmap with the correlation matrix
        between the variables.
        '''
        cols = [
            'deforestation_area_ha',
            'fires',
            'precipitation_mm',
            'farmland_area_ha'
        ]
        corr = self.__df[cols].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')   # Creates a heatmap with annotations
        plt.title('Correlation Matrix')
        path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f'✅ Correlation matrix saved in: {path}')


    def annual_charts(self):
        '''
        For each main variable, generates and saves a bar chart
        with the annual total (sum) over the time index.
        '''
        metrics = {
            'deforestation_area_ha': 'Deforestation area (ha)',
            'fires':                 'Fires',
            'precipitation_mm':      'Precipitation (mm)',
            'farmland_area_ha':     'Farmland area (ha/dia)'
        }
        # Ensure the index is a DatetimeIndex for resampling
        df = self.__df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_anual = df.resample('YE').sum()       # Resample the data to annual frequency and sum the values

        for col, label in metrics.items():
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                x=df_anual.index.year,
                y=df_anual[col]
            )
            plt.title(f'Yearly Total: {label}')
            plt.xlabel('Year')
            plt.ylabel(label)
            plt.xticks(rotation=45)
            fname = f'annual_{col}.png'
            path = os.path.join(OUTPUT_DIR, fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            print(f'✅ Annual chart for {label} saved in: {path}')


    def histograms(self):
        '''
        Generates histograms for each numeric variable and saves them in the output folder.
        '''
        num_cols = self.__df.select_dtypes(include='number').columns
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.__df[col])
            plt.title(f'Histogram for {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            path = os.path.join(OUTPUT_DIR, f'histogram_{col}.png')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            print(f'✅ Histogram for {col} saved in: {path}')


    def boxplots(self):
        '''
        Generates an individual boxplot for each numeric variable
        and saves it in the output folder.
        '''
        num_cols = self.__df.select_dtypes(include='number').columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.__df[col])
            plt.title(f'Boxplot for {col}')
            plt.xlabel(col)
            path = os.path.join(OUTPUT_DIR, f'boxplot_{col}.png')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            print(f'✅ Boxplot for {col} saved in: {path}')
            

    def run_analysis(self):
        '''
        Runs all EDA steps and informs the user.
        '''
        print('🔎 Starting exploratory data analysis...')
        self.numeric_statistics()
        self.correlation_matrix()
        self.annual_charts()
        self.histograms()
        self.boxplots()
        print('✅ Exploratory data analysis completed.')