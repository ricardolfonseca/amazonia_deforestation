# model/Forecast_amazonia.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from config import *
from controller.controller import *

import lightgbm as lgb
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


class Forecast_amazonia:
    '''
    Three rounds of regression forecasting for Amazon deforestation:
      1) Baseline models
      2) Tuned hyperparameters on normalized data
      3) Advanced methods (early stopping, CV, pipelines)
    '''

    def __init__(self, test_size):
        self.__test_size = test_size
        self.__dataset_path = DATASET_CLEAN
        self.__round_results = {}                        # Dictionary to store results for each round of train and test


    def plot_and_save_metrics(self, round_name, metrics):
        '''
        Generate a bar plot showing MAE, RMSE, and R² for each model,
        and save the figure to the forecast output directory.
        Parameters:
        - round_name: string identifying the training round (e.g. "Round 1")
        - metrics: dictionary with model evaluation results
        '''
        models = list(metrics.keys())                    # Get the list of model names from the dictionary keys
        # Extract metric values for each model
        mae   = [metrics[m]['MAE'] for m in models]        
        rmse  = [metrics[m]['RMSE'] for m in models]
        r2    = [metrics[m]['R2'] for m in models]

        x = range(len(models))                           # Create a numeric range to position each model on the x-axis
        plt.figure(figsize=(10, 6))
        plt.bar([p - 0.2 for p in x], mae,   width=0.2, label='MAE')    # Plot MAE bars slightly to the left
        plt.bar(x, rmse,  width=0.2, label='RMSE')       # Plot RMSE bars in the center
        plt.bar([p + 0.2 for p in x], r2,    width=0.2, label='R²')     # Plot R² bars slightly to the right
        plt.xticks(x, models, rotation=30)
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.title(f'{round_name} Metrics Comparison')
        plt.legend()
        plt.tight_layout()

        os.makedirs(OUTPUT_FORECAST_DIR, exist_ok=True)  # Create the output directory if it doesn't exist
        filename = f'{round_name.lower().replace(' ', '_')}_metrics.png'
        path = os.path.join(OUTPUT_FORECAST_DIR, filename)
        plt.savefig(path)
        plt.close()
        print(f'✅ {round_name} plot saved to {path}')


    def save_and_plot_monthly(self, round_name, df_daily):
        '''
        1) Aggregate df_daily by month-end
        2) Reset index to 'date' column, save CSV
        3) Plot bar (actual) + line (each pred) chart, using Month-Year ticks
        '''
        # 1) Aggregate by month
        df_monthly = aggregate_monthly_df(df_daily)

        # 2a) Reset index → 'date' column (keep as datetime for plotting)
        df_monthly = df_monthly.reset_index().rename(columns={'index': 'date'})

        # 2b) For the CSV, write full YYYY‐MM‐DD
        df_to_save = df_monthly.copy()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')

        os.makedirs(OUTPUT_FORECAST_DIR, exist_ok=True)
        csv_name = f'{round_name.lower().replace(' ', '_')}_monthly.csv'
        csv_path = os.path.join(OUTPUT_FORECAST_DIR, csv_name)
        df_to_save.to_csv(csv_path, index=False)
        print(f'✅ {round_name} monthly CSV saved to {csv_path}')

        # 3) Plot bar + lines, using actual datetime values
        plt.figure(figsize=(12, 6))

        # Bar chart for actual deforestation
        plt.bar(df_monthly['date'], df_monthly['deforestation_ha'],
                width=20,  # use a width in days (e.g. 20) so bars fill the month
                color='lightgray',
                label='Actual')

        # Overlay each prediction as a line
        for col in df_monthly.columns:
            if col in ('date', 'deforestation_ha'):
                continue
            plt.plot(df_monthly['date'], df_monthly[col],
                    marker='o', linewidth=2, label=col)

        # 3a) Use Month‐Year format on x‐axis
        ax = plt.gca()
        # Major locator: once every month
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        # Formatter: 'Aug‐2016', etc.
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Month')
        plt.ylabel('Total Deforestation (ha)')
        plt.title(f'{round_name}: Actual vs Predicted (Monthly)')
        plt.legend()
        plt.tight_layout()

        png_name = f'{round_name.lower().replace(' ', '_')}_monthly_plot.png'
        png_path = os.path.join(OUTPUT_FORECAST_DIR, png_name)
        plt.savefig(png_path)
        plt.close()
        print(f'✅ {round_name} monthly plot saved to {png_path}')


    def save_metrics_per_model(self):
        '''
        For each of the three model-families (LightGBM, Lasso, MLP),
        collect MAE, RMSE and R² across rounds into its own DataFrame,
        print it, and save as CSV under OUTPUT_FORECAST_DIR.
        '''
        # mapping from family label → function that identifies its entries
        families = {
            'lightgbm': lambda name: 'LightGBM' in name,
            'lasso':    lambda name: name.startswith('Lasso'),
            'mlp':      lambda name: 'MLP' in name
        }

        os.makedirs(OUTPUT_FORECAST_DIR, exist_ok=True)

        # define how to identify models belonging to each family based on their name
        for fam_label, match_fn in families.items():
            # build one dict per round
            data = {}
            for round_name, results in self.__round_results.items():
                # pick the one model in this round that matches
                candidates = [m for m in results if match_fn(m)]
                if not candidates:
                    continue
                model_name = candidates[0]                # only 1 match per round
                data[round_name] = results[model_name]    # store metrics for this round

            # DataFrame: index=rounds, columns=metrics
            df = pd.DataFrame(data).T  # transpose so rounds are rows
            # ensure column order
            df = df[['MAE', 'RMSE', 'R2']]

            # print to console
            print(f'\n--- {fam_label.upper()} metrics across rounds ---')
            print(df)

            # save to CSV
            fname = f'{fam_label}_metrics_across_rounds.csv'
            path  = os.path.join(OUTPUT_FORECAST_DIR, fname)
            df.to_csv(path)
            print(f'✅ Saved metrics for {fam_label.upper()} to {path}')


    def train_test_round_one(self):
        '''
        Round 1: Baseline models on raw features.
        '''
        print('\n---- Round 1: Baseline Models -----')
         # Prepare the dataset (features and target), split into train and test sets
        x_train, x_test, y_train, y_test, test_index = prepare_data_for_ml(
            self.__dataset_path, self.__test_size
        )

        results = {}                                      # Store evaluation metrics for each model

        # --- LightGBM ---
        print('\n-- LightGBM --')
        model_lgb = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)    # Gradient boosting model, n_jobs=-1 - use all CPU cores; verbose=-1 - suppress training logs
        model_lgb.fit(x_train, y_train)
        y_pred_lgb = model_lgb.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_lgb   = mean_squared_error(y_test, y_pred_lgb)
        rmse_lgb  = sqrt(mse_lgb)
        mae_lgb   = mean_absolute_error(y_test, y_pred_lgb)
        r2_lgb    = r2_score(y_test, y_pred_lgb)
        print(f'MSE:  {mse_lgb:.4f}\nRMSE: {rmse_lgb:.4f}\nMAE:  {mae_lgb:.4f}')
        results['LightGBM'] = {'MAE': mae_lgb, 'RMSE': rmse_lgb, 'R2': r2_lgb}   # store results

        # --- Lasso ---
        print('\n-- Lasso --')
        model_lasso = Lasso(alpha=0.01, max_iter=10_000)            # Linear model with L1 regularization,  alpha=0.01 - light L1 regularization; max_iter=10_000 - allow enough iterations for convergence
        model_lasso.fit(x_train, y_train)
        y_pred_lasso = model_lasso.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_lasso   = mean_squared_error(y_test, y_pred_lasso)      
        rmse_lasso  = sqrt(mse_lasso)
        mae_lasso   = mean_absolute_error(y_test, y_pred_lasso)
        r2_lasso    = r2_score(y_test, y_pred_lasso)
        print(f'MSE:  {mse_lasso:.4f}\nRMSE: {rmse_lasso:.4f}\nMAE:  {mae_lasso:.4f}')
        results['Lasso'] = {'MAE': mae_lasso, 'RMSE': rmse_lasso, 'R2': r2_lasso}    # store results

        # --- MLPRegressor ---
        print('\n-- MLPRegressor --')                               # MLP with two hidden layers (64, 32); max 500 iterations
        model_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model_mlp.fit(x_train, y_train)
        y_pred_mlp = model_mlp.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_mlp   = mean_squared_error(y_test, y_pred_mlp)           
        rmse_mlp  = sqrt(mse_mlp)
        mae_mlp   = mean_absolute_error(y_test, y_pred_mlp)
        r2_mlp    = r2_score(y_test, y_pred_mlp)
        print(f'MSE:  {mse_mlp:.4f}\nRMSE: {rmse_mlp:.4f}\nMAE:  {mae_mlp:.4f}')
        results['MLP'] = {'MAE': mae_mlp, 'RMSE': rmse_mlp, 'R2': r2_mlp}        # store results

        # Store metrics
        self.__round_results['Round 1'] = results

        # Build day‐level DataFrame using test_index
        df_daily = pd.DataFrame({
            'deforestation_ha': np.expm1(y_test),
            'pred_LightGBM':   np.expm1(y_pred_lgb),
            'pred_Lasso':      np.expm1(y_pred_lasso),
            'pred_MLP':        np.expm1(y_pred_mlp)
        }, index=test_index)

        # Plot & save monthly view
        self.save_and_plot_monthly('Round 1', df_daily)

        # Save metrics plot
        self.plot_and_save_metrics('Round 1', results)


    def train_test_round_two(self):
        '''
        Round 2: Tuned hyperparameters on normalized features.
        '''
        print('\n----- Round 2: Tuned + Normalized -----')
        x_train, x_test, y_train, y_test, test_index = prepare_data_for_ml(
            self.__dataset_path, self.__test_size
        )

        x_train_n, x_test_n = normalize_features(x_train, x_test)

        results = {}

        # --- LightGBM (tuned) ---
        print('\n-- LightGBM (tuned) --')
        model_lgb = lgb.LGBMRegressor(
            n_estimators=300,                            # Number of boosting rounds (trees) 
            learning_rate=0.1,                           # Step size shrinkage to prevent overfitting
            max_depth=12,                                # Maximum depth of each tree
            random_state=42,                             # Ensures reproducibility
            n_jobs=-1,                                   # Use all available cores
            verbose=-1                                   # Suppress output during training
        )
        model_lgb.fit(x_train_n, y_train)
        y_pred_lgb = model_lgb.predict(x_test_n)

        # Evaluate predictions using common regression metrics
        mse_lgb   = mean_squared_error(y_test, y_pred_lgb)
        rmse_lgb  = sqrt(mse_lgb)
        mae_lgb   = mean_absolute_error(y_test, y_pred_lgb)
        r2_lgb    = r2_score(y_test, y_pred_lgb)
        print(f'MSE:  {mse_lgb:.4f}\nRMSE: {rmse_lgb:.4f}\nMAE:  {mae_lgb:.4f}')
        results['LightGBM (tuned)'] = {'MAE': mae_lgb, 'RMSE': rmse_lgb, 'R2': r2_lgb}        # store results

        # --- Lasso (tuned) ---
        print('\n-- Lasso (tuned) --')
        model_lasso = Lasso(alpha=0.1, max_iter=20_000)        # stronger regularization (alpha=0.1) and higher max iterations (20k)
        model_lasso.fit(x_train_n, y_train)
        y_pred_lasso = model_lasso.predict(x_test_n)

        # Evaluate predictions using common regression metrics
        mse_lasso   = mean_squared_error(y_test, y_pred_lasso)
        rmse_lasso  = sqrt(mse_lasso)
        mae_lasso   = mean_absolute_error(y_test, y_pred_lasso)
        r2_lasso    = r2_score(y_test, y_pred_lasso)
        print(f'MSE:  {mse_lasso:.4f}\nRMSE: {rmse_lasso:.4f}\nMAE:  {mae_lasso:.4f}')
        results['Lasso (tuned)'] = {'MAE': mae_lasso, 'RMSE': rmse_lasso, 'R2': r2_lasso}    # store results

        # --- MLPRegressor (tuned) ---
        print('\n-- MLPRegressor (tuned) --')
        model_mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),                  # Three hidden layers with decreasing size (additional layers to capture more complext patterns)
            max_iter=1000,                                     # Increased iterations to improve convergence
            random_state=42
        )
        model_mlp.fit(x_train_n, y_train)
        y_pred_mlp = model_mlp.predict(x_test_n)

        # Evaluate predictions using common regression metrics
        mse_mlp   = mean_squared_error(y_test, y_pred_mlp)
        rmse_mlp  = sqrt(mse_mlp)
        mae_mlp   = mean_absolute_error(y_test, y_pred_mlp)
        r2_mlp    = r2_score(y_test, y_pred_mlp)
        print(f'MSE:  {mse_mlp:.4f}\nRMSE: {rmse_mlp:.4f}\nMAE:  {mae_mlp:.4f}')
        results['MLP (tuned)'] = {'MAE': mae_mlp, 'RMSE': rmse_mlp, 'R2': r2_mlp}        # store results

        # Store metrics
        self.__round_results['Round 2'] = results

        # Build day‐level DataFrame using test_index
        df_daily = pd.DataFrame({
            'deforestation_ha': np.expm1(y_test),
            'pred_LightGBM':    np.expm1(y_pred_lgb),
            'pred_Lasso':       np.expm1(y_pred_lasso),
            'pred_MLP':         np.expm1(y_pred_mlp)
        }, index=test_index)

        # Plot & save monthly view
        self.save_and_plot_monthly('Round 2', df_daily)

        # Save metrics plot
        self.plot_and_save_metrics('Round 2', results)


    def train_test_round_three(self):
        '''
        Round 3: Advanced workflows (early stopping, CV, pipelines).
        '''
        print('\n----- Round 3: Advanced Methods -----')
        x_train, x_test, y_train, y_test, test_index = prepare_data_for_ml(
            self.__dataset_path, self.__test_size
        )

        results = {}

        # --- LightGBM Early Stopping ---
        print('\n-- LightGBM Early Stopping --')
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1000,                    # Max number of trees (will stop earlier if no improvement)
            learning_rate=0.05,                   # Smaller learning rate for better generalization
            max_depth=10,                         # Control tree complexity
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        # Includes validation of RMSE, if no improvements in 50 rounds, stops
        model_lgb.fit(
            x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
        y_pred_lgb = model_lgb.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_lgb   = mean_squared_error(y_test, y_pred_lgb)
        rmse_lgb  = sqrt(mse_lgb)
        mae_lgb   = mean_absolute_error(y_test, y_pred_lgb)
        r2_lgb    = r2_score(y_test, y_pred_lgb)
        print(f'MSE:  {mse_lgb:.4f}\nRMSE: {rmse_lgb:.4f}\nMAE:  {mae_lgb:.4f}')
        results['LightGBM ES'] = {'MAE': mae_lgb, 'RMSE': rmse_lgb, 'R2': r2_lgb}

        # --- LassoCV (TimeSeriesSplit) ---
        print('\n-- LassoCV (TimeSeriesSplit) --')
        tscv = TimeSeriesSplit(n_splits=5)
        model_lasso_cv = LassoCV(
            alphas=[0.001, 0.01, 0.1, 1.0],        # Try multiple regularization strengths
            cv=tscv,                               # Time-aware cross-validation to preserve temporal order
            max_iter=20_000,
            random_state=42
        )
        model_lasso_cv.fit(x_train, y_train)
        print(f'Best alpha: {model_lasso_cv.alpha_:.4f}')
        y_pred_lasso = model_lasso_cv.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_lasso   = mean_squared_error(y_test, y_pred_lasso)
        rmse_lasso  = sqrt(mse_lasso)
        mae_lasso   = mean_absolute_error(y_test, y_pred_lasso)
        r2_lasso    = r2_score(y_test, y_pred_lasso)
        print(f'MSE:  {mse_lasso:.4f}\nRMSE: {rmse_lasso:.4f}\nMAE:  {mae_lasso:.4f}')
        results['LassoCV'] = {'MAE': mae_lasso, 'RMSE': rmse_lasso, 'R2': r2_lasso}

        # --- MLP Pipeline (Scaler + Early Stopping) ---
        print('\n-- MLP Pipeline (Scaler + ES) --')
        pipe = Pipeline([
            ('scaler',  StandardScaler()),                # Normalize features before training MLP
            ('mlp',     MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                early_stopping=True,                      # Stop training if validation score stops improving
                validation_fraction=0.1,                  # Use 10% of training data for validation
                n_iter_no_change=20,                      # Stop after 20 rounds with no change
                max_iter=2000,
                random_state=42
            ))
        ])
        pipe.fit(x_train, y_train)
        y_pred_mlp = pipe.predict(x_test)

        # Evaluate predictions using common regression metrics
        mse_mlp   = mean_squared_error(y_test, y_pred_mlp)
        rmse_mlp  = sqrt(mse_mlp)
        mae_mlp   = mean_absolute_error(y_test, y_pred_mlp)
        r2_mlp    = r2_score(y_test, y_pred_mlp)
        print(f'MSE:  {mse_mlp:.4f}\nRMSE: {rmse_mlp:.4f}\nMAE:  {mae_mlp:.4f}')
        results['MLP Pipeline'] = {'MAE': mae_mlp, 'RMSE': rmse_mlp, 'R2': r2_mlp}

        # Store metrics
        self.__round_results['Round 3'] = results

        # Build day‐level DataFrame using test_index
        df_daily = pd.DataFrame({
            'deforestation_ha': np.expm1(y_test),
            'pred_LightGBM':    np.expm1(y_pred_lgb),
            'pred_Lasso':       np.expm1(y_pred_lasso),
            'pred_MLP':         np.expm1(y_pred_mlp)
        }, index=test_index)

        # Plot & save monthly view
        self.save_and_plot_monthly('Round 3', df_daily)

        # Save metrics plot
        self.plot_and_save_metrics('Round 3', results)
