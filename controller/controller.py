# controller/controller.py

from config import *

import pandas as pd
import numpy as np
import calendar
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import HeatMap

from sklearn.preprocessing import StandardScaler


def merge_datasets():
    '''
    Reads the four datasets (deforestation, fires, precipitation, and pasture),
    merges them into a single daily DataFrame, and saves it with the 'date' column as the time index.
    '''
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
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(columns=['year', 'month', 'day'], inplace=True)

    # 8) Sort and save the final CSV
    df = df.sort_values('date').set_index('date')
    df.to_csv(DATASET_MERGED, index=True)
    print(f'✅ Complete dataset saved at: {DATASET_MERGED}')

    return df


def prepare_data_for_ml(dataset_path, test_fraction):
    '''
    Load the cleaned Amazonia dataset, create features & target,
    and split it into temporal train/test sets. Returns:
       x_train, x_test, y_train (Series), y_test, test_index (DatetimeIndex of x_test/y_test)
    '''

    # 1) Read the CSV file and set the 'date' column as a DatetimeIndex, which is crucial for temporal operations like lags and rolling windows
    df = pd.read_csv(dataset_path, parse_dates=['date'], index_col='date')

    # 2) Transform target using log1p to handle zero values, reduce skew from large deforestation spikes, 
    # and make the distribution more suitable for modeling; apply expm1() later to recover original scale
    df['y'] = np.log1p(df['deforestation_area_ha'])

    # 3) Create lag features (1, 7, 30 days) to give the model memory of recent deforestation patterns
    # helps capture short-term trends and temporal dependencies
    for lag in (1, 7, 30):
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # 4) Add rolling mean features (7 and 30 days) to capture recent trends and smooth out daily fluctuations
    # helps the model understand short- and medium-term deforestation patterns
    for window in (7, 30):
        df[f'rolling_mean_{window}'] = df['y'].rolling(window).mean()

    # 5) Seasonal indicators
    df['month']       = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['year']        = df.index.year

    # 6) Transform fires and precipitation using log1p to handle zero values and reduce skew
    df['log_focos']        = np.log1p(df['fires'])
    df['log_precipitacao'] = np.log1p(df['precipitation_mm'])

    # 7) Drop missing values
    df = df.drop(columns=['deforestation_area_ha']).dropna()

    # 8) Select explanatory variables (features) for training and the target variable (y)
    # X will be used as input for the model, and y is the value to be predicted (log of deforestation)
    feature_cols = [
        'log_focos', 'log_precipitacao', 'pasture_area_ha',
        'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30',
        'month', 'day_of_year'
    ]
    X = df[feature_cols]
    y = df['y']

    # 9) Temporal split (sequential split of rows) of the dataset into training and test sets
    split = int(len(df) * (1 - test_fraction))              # Position of the row number to split the dataset
    x_train = X.iloc[:split]                                # From the start to the split position
    x_test  = X.iloc[split:]                                # From the split position to the end    
    y_train = y.iloc[:split]                                # Corresponding target values for training  
    y_test  = y.iloc[split:]                                # Corresponding target values for testing

    # Capture the actual DatetimeIndex for the test split, which allows us to track the dates of the test set and plot results
    test_index = df.index[split:]

    print(f'✅ Data prepared: {len(x_train)} train / {len(x_test)} test\n')
    return x_train, x_test, y_train, y_test, test_index


def normalize_features(x_train, x_test):
    '''
    Fit a StandardScaler using x_train (to avoid data leakage), then transform both x_train and x_test 
    so that all features have mean 0 and std 1; returns normalized arrays for model training/testing
    '''
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)
    return x_tr, x_te


def aggregate_monthly_df(df):
    '''
    Given a DataFrame with a DatetimeIndex and numeric columns,
    group by calendar month (month-end) and sum all numeric values in each month.
    Returns a new DataFrame indexed at month-end dates.
    '''
    df = df.copy()                                                  # Avoids modifying the original DataFrame    
    df.index = pd.to_datetime(df.index)                             # Ensures the index is a DatetimeIndex  
    monthly = df.groupby(pd.Grouper(freq='ME')).sum()               # Group by month-end
    return monthly


def plot_metrics_plotly(metrics_dict, title):
    """
    metrics_dict: 
      { 'LightGBM': {'MAE':…, 'RMSE':…, 'R2':…}, 
        'Lasso':    {'MAE':…, 'RMSE':…, 'R2':…}, 
        'MLP':      {'MAE':…, 'RMSE':…, 'R2':…} }
    Builds a DataFrame, then uses px.bar(...) with barmode='group'.
    """
    dfm = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Model'})
    # dfm columns = ['Model', 'MAE', 'RMSE', 'R2']
    fig = px.bar(
        dfm,
        x='Model',
        y=['MAE', 'RMSE', 'R2'],
        barmode='group',
        title=title,
        labels={'value': 'Metric value', 'variable': 'Metric'}
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_deforestation(round_name):
    """
    Reads: OUTPUT_FORECAST_DIR / f'{round_name.lower().replace(" ", "_")}_monthly.csv'
    Expects columns: ['date','deforestation_ha','pred_LightGBM','pred_Lasso','pred_MLP'].
    Plots a bar (Actual) + lines (pred_LightGBM, pred_Lasso, pred_MLP), with x = month end.
    """
    csv_fname = f'{round_name.lower().replace(" ", "_")}_monthly.csv'
    csv_path = os.path.join(OUTPUT_FORECAST_DIR, csv_fname)
    if not os.path.exists(csv_path):
        st.warning(f'No monthly CSV found for {round_name} at:\n{csv_path}')
        return

    dfm = pd.read_csv(csv_path, parse_dates=['date'])

    fig = go.Figure()
    # Bar for actual deforestation
    fig.add_trace(
        go.Bar(
            x=dfm['date'],
            y=dfm['deforestation_ha'],
            name='Actual',
            marker_color='lightgray',
        )
    )
    # Lines for each prediction column
    for col in ['pred_LightGBM', 'pred_Lasso', 'pred_MLP']:
        fig.add_trace(
            go.Scatter(
                x=dfm['date'],
                y=dfm[col],
                mode='lines+markers',
                name=col.replace('pred_', 'Pred: '),
            )
        )

    fig.update_layout(
        title=f'{round_name}: Actual vs Predicted (Monthly)',
        xaxis_title='',
        yaxis_title='Total Deforestation (ha)',
        legend_title='Legend',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Format x-axis as Month-Year, rotated 45°
    fig.update_xaxes(tickformat='%b-%Y', tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)  # Enables caching of the DataFrame for faster access
def load_aggregated_coordinates():
    '''
    Reads the Parquet file containing aggregated deforestation points
    and returns a pandas DataFrame with columns: ['year', 'lat', 'lon', 'count'].
    '''
    # Ensure the file exists before attempting to read it
    if not os.path.exists(OUTPUT_AGG_COORDINATES):
        raise FileNotFoundError(f'Parquet not found at: {OUTPUT_AGG_COORDINATES}')
    return pd.read_parquet(OUTPUT_AGG_COORDINATES)


@st.cache_data(show_spinner=False)  # Enables caching of the generated HTML for faster updates
def generate_heatmap_html(start_year: int, end_year: int, df_agg: pd.DataFrame) -> str:
    '''
    Creates a Folium map with a HeatMap overlay for the specified year range.
    - Filters df_agg to include only rows where 'year' is between start_year and end_year.
    - Converts the filtered rows into a list of [lat, lon, weight] for the HeatMap.
    - Centers the map on the Amazon region and adds the HeatMap layer.
    - Returns the HTML <iframe> representation of the map.
    '''
    # 1. Filter the DataFrame by the selected year range
    df_filtered = df_agg[(df_agg['year'] >= start_year) & (df_agg['year'] <= end_year)]
    
    # 2. Prepare a list of [latitude, longitude, count] for the HeatMap plugin
    points = df_filtered[['lat', 'lon', 'count']].values.tolist()
    
    # 3. Create a Folium map centered on the Amazon basin
    folium_map = folium.Map(location=[-5, -60], zoom_start=4)
    
    # 4. Define a color gradient (white → red)
    gradient = {'0': 'white', '1': 'red'}
    
    # 5. Add the HeatMap layer to the map
    HeatMap(
        data=points,
        radius=12,
        blur=15,
        min_opacity=0.2,
        max_opacity=0.8,
        gradient=gradient
    ).add_to(folium_map)
    
    # 6. Return the HTML <iframe> representation generated by Folium
    return folium_map._repr_html_()
