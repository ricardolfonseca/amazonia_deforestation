# view.py

import os
import pandas as pd

from model.Deforestation_amazonia import *
from model.Fires_Amazonia import *
from model.Precipitation_amazonia import *
from model.Pasture_amazonia import *
from controller.controller import *
from model.Data_cleaner import *
from model.Eda import *
from model.Forecast_amazonia import *
from config import *


def main():

    # 1) Final cleaned dataset
    if os.path.exists(DATASET_CLEAN):
        print(f'✅ Cleaned dataset already exists: {DATASET_CLEAN}')
        df_clean = pd.read_csv(DATASET_CLEAN, parse_dates=['date'], index_col='date')
        print(df_clean.head(10))
    else:
        # 2) Merged dataset
        if os.path.exists(DATASET_MERGED):
            print(f'✅ Complete dataset already exists: {DATASET_MERGED}')
            df_merged = pd.read_csv(DATASET_MERGED, parse_dates=['date'], index_col='date')
            print(df_merged.head(10))
        else:
            # 3) Individual datasets
            print('=== Creating/checking individual datasets ===')

            # Deforestation
            if os.path.exists(DF_DEFORESTATION):
                print(f'✅ Deforestation already exists: {DF_DEFORESTATION}')
                df_def = pd.read_csv(DF_DEFORESTATION)
            else:
                print('=== Processing deforestation ===')
                df_def = Deforestation_amazonia().process_shapefile()
            print(df_def.head())

            # Fires
            if os.path.exists(DF_FIRES):
                print(f'✅ Fires already exists: {DF_FIRES}')
                df_fires = pd.read_csv(DF_FIRES)
            else:
                print('=== Processing fires ===')
                df_fires = Fires_Amazonia().process_shapefile()
            print(df_fires.head())

            # Precipitation
            if os.path.exists(DF_PRECIPITATION):
                print(f'✅ Precipitation already exists: {DF_PRECIPITATION}')
                df_prec = pd.read_csv(DF_PRECIPITATION)
            else:
                print('=== Processing precipitation ===')
                df_prec = Precipitation_amazonia().prepare_dataset()
            print(df_prec.head())

            # Pasture
            if os.path.exists(DF_PASTURE):
                print(f'✅ pasture already exists: {DF_PASTURE}')
                df_farm = pd.read_csv(DF_PASTURE)
            else:
                print('=== Processing pasture ===')
                df_farm = Pasture_amazonia().process()
            print(df_farm.head())

            # Final merge
            df_merged = merge_datasets()
            print(f'✅ Complete dataset saved at: {DATASET_MERGED}')
            print(df_merged.head(10))

        # 4) Cleaning
        print('=== Cleaning the complete dataset ===')
        cleaner = Data_cleaner(df_merged)
        cleaner.remove_duplicates()

        # Extend pasture until Mar/2025
        cleaner.extend_pasture(2025, 3)

        # Filter by temporal interval with deforestation data
        cleaner.filter_by_time_interval(2016, 8, 2025, 3)

        # Now replace remaining nulls (in other columns) with zero
        cleaner.replace_nulls_with_zero()

        # Save the cleaned DataFrame and return the first 10 rows
        df_clean = cleaner.save_and_get_df()
        print(df_clean.head(10))


    # --- EDA - Exploratory Data Analysis ---
    eda = Eda()
    eda.run_analysis()


    # --- Machine Learning ---

    # Instantiate the predictor.
    # The dataset will be split by time into training and test sets with a 20% test size.
    predictor = Forecast_amazonia(test_size=0.2)

    # ────────────────────────────────────────────────────────────────────────────
    # Round 1: Baseline models on raw features
    #   • LightGBM with default settings
    #   • Lasso (α=0.01)
    #   • MLPRegressor (64-32 hidden layers)
    #   – No normalization, no hyperparameter tuning
    # ────────────────────────────────────────────────────────────────────────────
    predictor.train_test_round_one()

    # ────────────────────────────────────────────────────────────────────────────
    # Round 2: Tuned hyperparameters on normalized data
    #   • LightGBM (n_estimators=300, learning_rate=0.1, max_depth=12)
    #   • Lasso (α=0.1)
    #   • MLPRegressor (128-64-32 hidden layers, more iterations)
    #   – Features are scaled via StandardScaler before training
    # ────────────────────────────────────────────────────────────────────────────
    predictor.train_test_round_two()

    # ────────────────────────────────────────────────────────────────────────────
    # Round 3: Advanced workflows
    #   • LightGBM with early stopping (1,000 trees, rmse callback)
    #   • LassoCV with TimeSeriesSplit (automatically selects α)
    #   • MLP pipeline combining StandardScaler + MLPRegressor with early stopping
    #   – Combines best practices: callbacks, cross-validation, and pipelines
    # ────────────────────────────────────────────────────────────────────────────
    predictor.train_test_round_three()

    # now build & save the comparison table
    predictor.save_metrics_per_model()

    print()

    print("\n🏁 End of pipeline.\n")