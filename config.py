# config.py

import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main directories
DATA_DIR       = os.path.join(BASE_DIR, 'model', 'data')
RAW_DATA_DIR   = os.path.join(DATA_DIR, 'raw')
MODELS_DIR     = os.path.join(DATA_DIR, 'model')
CLEAN_DATA     = os.path.join(DATA_DIR, 'clean')

# Specific subdirectories
# Shapes and shapefiles
SHAPEFILE_DIR     = os.path.join(BASE_DIR, 'model', 'data', 'shapefile')
SHAPEFILE_ALERTS = os.path.join(SHAPEFILE_DIR, 'deter-amz-deter-public.shp')
SHAPEFILE_AMAZONIA   = os.path.join(SHAPEFILE_DIR, 'brazilian_legal_amazon.shp')

# Raw data directories
DEFORESTATION_DIR = os.path.join(RAW_DATA_DIR, 'deforestation')
FIRES_DIR = os.path.join(RAW_DATA_DIR, 'fires')
PRECIPITATION_DIR   = os.path.join(RAW_DATA_DIR, 'precipitation')
PASTURE_DIR       = os.path.join(RAW_DATA_DIR, 'pasture')

# Individual datasets
DF_DEFORESTATION = os.path.join(DEFORESTATION_DIR, 'deforestation_amazonia_daily.csv')
DF_FIRES = os.path.join(FIRES_DIR, 'fires_amazonia_daily.csv')
DF_PRECIPITATION = os.path.join(PRECIPITATION_DIR, 'precipitation_amazonia_daily.csv')
DF_PASTURE = os.path.join(PASTURE_DIR, 'pasture_amazonia_yearly.csv')

# Merged dataset
DATASET_MERGED = os.path.join(RAW_DATA_DIR, 'amazonia_complete_dataset.csv')

# Clean dataset
DATASET_CLEAN = os.path.join(CLEAN_DATA, 'amazonia_dataset.csv')

# Outputs
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
OUTPUT_EDA_DIR = os.path.join(OUTPUT_DIR, 'EDA')
OUTPUT_FORECAST_DIR = os.path.join(OUTPUT_DIR, 'forecast')

# Earth Engine credentials
EE_CREDENTIALS = os.getenv(
    'EE_CREDENTIALS',
    os.path.join(BASE_DIR, 'assets', 'earthengine-project.json')    # Here you must provide the path to your Earth Engine credentials file
)


# Buckets Google Cloud
BUCKET_PRECIPITATION = 'amazonia-chuva-dados'                       # Replace with your bucket name, if needed

# Years of interest
START_YEAR = 2016
END_YEAR   = 2025