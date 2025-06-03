# 🌳 Amazon Deforestation Analysis

> The Amazon is not only the lungs of our planet, it's also one of the most vital frontiers in the fight against climate change.

This repository contains the final project for my MBA in Data Science, focused on the deforestation of the Brazilian Legal Amazon.  
The Amazon rainforest plays a fundamental role in global climate regulation, biodiversity conservation, and indigenous protection. Monitoring and mitigating its deforestation is not just a national concern — it’s a global responsibility.



## 💡 Why this project matters

Deforestation in the Amazon is a complex phenomenon driven by multiple environmental and human factors. This project aims to integrate and analyze large volumes of data to generate actionable insights for predicting and understanding forest loss.  
It aligns with the broader mission of applying data science for good — by using advanced technology to support environmental sustainability.


## 🛠️ Tools & Technologies Applied

This project brings together several key subjects from my MBA in Data Science:

- **Big Data & Cloud**: Use of Google Earth Engine (EE) and Google Cloud Storage (GCS) to process large-scale geospatial and environmental datasets
- **Python Programming**: Full data analysis pipeline using Python libraries like Pandas, GeoPandas, NumPy, Matplotlib and more
- **APIs**: Integration with Earth Engine API and GCS API for automated extraction and export
- **Statistics and Predictive Analytics**: Statistical analysis to identify patterns and support forecasts
- **Machine Learning & AI**: Preparation for predictive modeling using Random Forest, LightGBM, and XGBoost
- **Geospatial Analysis**: Combining shapefiles, raster data, and temporal-spatial trends
- **Data Visualization**: Interactive visualization using Streamlit (to be included)


## 🤖 Machine Learning Forecasting

A key part of this project is the application of regression-based machine learning models to forecast deforestation trends in the Brazilian Amazon.

Implemented in `model/Forecast_amazonia.py`, the forecasting workflow is structured into three progressive rounds:

1. **Baseline models**: LightGBM, Lasso Regression, and Multi-layer Perceptron (MLP) using default parameters  
2. **Tuned models on normalized data**: Hyperparameter tuning and feature scaling to improve accuracy  
3. **Advanced workflows**: Integration of early stopping, cross-validation (TimeSeriesSplit), and pipelines (e.g., MLP + StandardScaler)

Each model is evaluated using standard regression metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Predictions and metrics are plotted and exported in monthly aggregates, offering visual insight into performance and temporal trends.

The final goal is to build robust models capable of anticipating forest loss based on historical patterns and environmental drivers.


## 📌 Objectives

- Combine multiple data sources on deforestation, fire outbreaks, precipitation, population, and vegetation exploitation.
- Explore temporal and spatial patterns of forest loss.
- Create ready-to-use datasets for machine learning models to predict deforestation.
- Share insights on Amazon forest degradation based on public and official data.
- Predict deforestation changes based on current data.


## 🏃 How to run the project

- Simply run `main.py` on your local IDE.


## 📁 Project Structure

```
📦 amazonia-deforestation
├── assets/                  # (not included) Earth Engine credentials
├── config.py                # Global project parameters
├── main.py                  # Main execution script
├── controller/              # Data cleaning and integration functions
├── model/                   # Data and Classes
├── view/                    # Visualizations and reporting
├── data/                    # Local data (not versioned)
├── README.md                # This file
├── requirements.txt         # Required libraries
```


## 🔧 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

> ⚠️ Note: To run Earth Engine scripts, you need to add your credentials in `assets/earthengine-project.json`. However, all data analyses can be executed with the available local datasets.


## 📊 Data Sources

- **Legal Amazon Shapefile**: [TerraBrasilis catalogue - Download](https://terrabrasilis.dpi.inpe.br/geonetwork/srv/eng/catalog.search#/metadata/d6289e13-c6f3-4103-ba83-13a8452d46cb)
- **DETER deforestation alerts**: [TerraBrasilis catalogue - Download](https://terrabrasilis.dpi.inpe.br/geonetwork/srv/por/catalog.search#/metadata/f2153c4a-915b-48a6-8658-963bdce7366c)
- **Fire outbreaks, precipitation, and pasture data**: Extracted using [Google Earth Engine (GEE)](https://earthengine.google.com/) and stored in Google Cloud Storage (GCS) bucket


## 🗂️ Optional Raw Files

To reproduce the full pipeline from raw geospatial data, you may optionally download the following files. These are **not required** if you're working directly with the cleaned and integrated datasets already provided in the repository.

- 🗺️ **Shapefile (DETER deforestation alerts)**:  
  [Download from Google Drive](https://drive.google.com/file/d/1ynOiSeX7aQWXz0BBhAEpOm9GMKxVGKBW/view?usp=sharing)

- 🔥 **GeoJSON (Fire outbreaks from FIRMS 2016–2025)**:  
  [Download from Google Drive](https://drive.google.com/file/d/1JtdgzR2VXMZ4hn3CpoNqMrDKm1d7C6BO/view?usp=sharing)

Once downloaded, place the files in the following local directories (create them if needed):

```
model/data/shapefile/
model/data/raw/fires/
```

> ⚠️ These files are ignored by Git and not versioned due to size limitations.


## 🚧 Roadmap

- ✅ Exploratory and spatial data analysis
- ✅ Integration of environmental variables: fires, rain and farming
- ✅ Data preparation for ML modeling
- ✅ Forecasting Amazon deforestation using Machine Learning
- 🔜 Interactive Dashboard and report


## 📬 Author

**Ricardo Fonseca**  
MBA in Data Science — Autónoma Academy  
[LinkedIn](https://www.linkedin.com/in/ricardolopesfonseca/)