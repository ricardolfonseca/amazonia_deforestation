# 🌳 Amazon Deforestation Analysis

> The Amazon is not only the lungs of our planet, it's also one of the most vital frontiers in the fight against climate change.

This repository contains the final project for my MBA in Data Science, focused on deforestation in the Brazilian Legal Amazon.  
The Amazon rainforest plays a fundamental role in global climate regulation, biodiversity conservation, and indigenous protection. Monitoring and mitigating its deforestation is not just a national concern—it’s a global responsibility.

## 💡 Why This Project Matters

Deforestation in the Amazon is a complex phenomenon driven by multiple environmental and human factors.  
By integrating and analyzing large volumes of data, this project generates actionable insights for predicting and understanding forest loss.  
It aligns with the broader mission of applying data science for good—using advanced technology to support environmental sustainability.

## 🛠️ Tools & Technologies Applied

This project brings together several key subjects from my MBA in Data Science:

- **Big Data & Cloud**  
  - Google Earth Engine (EE) for large‐scale geospatial extraction  
  - Google Cloud Storage (GCS) for intermediate data exports
- **Python Programming**  
  - Core libraries: Pandas, GeoPandas, NumPy, Matplotlib, Plotly, etc.
- **APIs**  
  - Earth Engine API & GCS API for automated data ingestion
- **Statistics & Predictive Analytics**  
  - EDA, correlation analysis, and time‐series exploration
- **Machine Learning & AI**  
  - LightGBM, Lasso Regression, and Multi‐Layer Perceptron (MLP)  
  - Structured as three progressively refined “rounds” of modeling
- **Geospatial Analysis**  
  - Shapefiles, raster layers, and temporal‐spatial pattern discovery
- **Data Visualization**  
  - Interactive Report built with Streamlit

## 🤖 Machine Learning Forecasting

A key part of this project is the application of regression‐based machine learning models to forecast deforestation trends in the Brazilian Amazon.

Implemented in `model/Forecast_amazonia.py`, the forecasting workflow is structured into three progressive rounds:

1. **Baseline models**  
   - LightGBM, Lasso Regression, and MLP using default parameters on raw features
2. **Tuned models on normalized data**  
   - Hyperparameter tuning and feature scaling to improve accuracy
3. **Advanced workflows**  
   - LightGBM with early stopping  
   - LassoCV with `TimeSeriesSplit`  
   - MLP inside a `Pipeline` with `StandardScaler` and early stopping

Each model is evaluated on a hold‐out set with standard regression metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Monthly aggregations of actuals vs. predictions are saved as CSVs (e.g., `round_1_monthly.csv`, `round_2_monthly.csv`, etc.) and visualized to show performance over time.

The final goal is to build robust models capable of anticipating forest loss based on historical patterns and environmental drivers.

## 📌 Objectives

- Combine multiple data sources on deforestation, fire outbreaks, precipitation, population, and land use
- Explore temporal and spatial patterns of forest loss
- Create cleaned, ready‐to‐use datasets for machine learning
- Share insights on Amazon forest degradation based on public and official data
- Predict near‐future deforestation trends to inform policy and conservation

## 🚀 Interactive Report (Streamlit)

The project now includes an [interactive Streamlit report](https://amazoniadeforestation-rlfonseca.streamlit.app/) (`app.py`) that presents:

1. **Introduction & Context**  
2. **Map of Legal Amazon Boundary**  
3. **Exploratory Data Analysis** (EDA)  
   - Numeric summaries, correlation heatmap, annual trends, histograms, boxplots  
4. **Machine Learning Forecasting**  
   - Three “Run Round” buttons (Baseline, Tuned + Normalized, Advanced)  
   - For each round:  
     - Side‐by‐side Plotly chart of MAE/RMSE/R² and a metrics table  
     - Full‐width Plotly monthly “Actual vs. Predicted” plot  
     - A brief 2–3 line summary of that round’s results  
   - Final “Show Final Recommendation” button displays a static summary card highlighting the best model.

**How to launch the dashboard:**  
1. Activate your Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. From the project root, run:
   ```bash
   streamlit run app.py
   ```
3. Your browser will open at `http://localhost:8501/`, where you can navigate through the sections and click each “Run Round” button to see results in real time.


## ⚙️ Project Structure

```text
amazonia_deforestation/
├── assets/                    # (not committed) Earth Engine credentials / supporting images
├── config.py                  # Global constants (paths, GCS buckets, etc.)
├── app.py                     # Streamlit dashboard entry point
├── controller/                # Helper functions for plotting and data integration
│   ├── __init__.py
│   └── controller.py
├── model/                     # Data classes and ML pipeline
│   ├── Forecast_amazonia.py
│   └── data/                  # Cleaned and raw data subfolders (not versioned)
├── view/                      # Additional visualizations / reporting (if any)
├── main.py                    # Legacy “run all steps” script (now superseded by app.py)
├── data/                      # Local datasets used for EDA & ML (not versioned)
├── README.md                  # This file
└── requirements.txt           # All required Python packages
```

## 🔧 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

> ⚠️ Note: To run Earth Engine scripts, you need valid EE credentials in `assets/earthengine-project.json`. However, the Streamlit dashboard can run entirely with the cleaned datasets already provided.

## 📊 Data Sources

- **Legal Amazon Shapefile**  
  TerraBrasilis catalogue — [Download link](https://terrabrasilis.dpi.inpe.br/geonetwork/srv/eng/catalog.search#/metadata/d6289e13-c6f3-4103-ba83-13a8452d46cb)
- **DETER Deforestation Alerts**  
  TerraBrasilis catalogue — [Download link](https://terrabrasilis.dpi.inpe.br/geonetwork/srv/por/catalog.search#/metadata/f2153c4a-915b-48a6-8658-963bdce7366c)
- **Fire Outbreaks, Precipitation, Pasture Data**  
  Extracted via Google Earth Engine and stored in a GCS bucket

## 🗂️ Optional Raw Files

If you want to reproduce the entire pipeline from raw geospatial data, you may download:

- 🗺️ **DETER Shapefile (raw alerts)**  
  [Google Drive link](https://drive.google.com/file/d/1ynOiSeX7aQWXz0BBhAEpOm9GMKxVGKBW/view?usp=sharing)
- 🔥 **GeoJSON Fire Outbreaks (FIRMS 2016–2025)**  
  [Google Drive link](https://drive.google.com/file/d/1JtdgzR2VXMZ4hn3CpoNqMrDKm1d7C6BO/view?usp=sharing)

Place them under the appropriate local folders:

```
model/data/shapefile/
model/data/raw/fires/
```

> These large raw files are ignored by Git (in `.gitignore`) due to size.

## 🚧 Roadmap

- ✅ Exploratory and spatial data analysis  
- ✅ Integration of environmental variables: fires, rainfall, land use  
- ✅ Data cleaning and ML dataset preparation  
- ✅ Forecasting Amazon deforestation with LightGBM, Lasso, and MLP  
- ✅ Interactive dashboard (Streamlit)  
- ✅ Add interactive map layer (deforestation over the years)  
- ✅ Deploy dashboard to a cloud service (Streamlit Cloud, Heroku, etc.)


## 📬 Author

**Ricardo Fonseca**  
MBA in Data Science — Autónoma Academy  
[LinkedIn](https://www.linkedin.com/in/ricardolopesfonseca/)
