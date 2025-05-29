# 🌳 Amazon Deforestation Analysis

This repository contains the final project for my MBA in Data Science, focused on the deforestation of the Brazilian Legal Amazon. It integrates multiple environmental, geospatial, and socioeconomic datasets to analyze patterns and prepare for future machine learning modeling.

---

## 📌 Objectives

- Combine multiple data sources on deforestation, fire outbreaks, precipitation, population, and vegetation exploitation.
- Explore temporal and spatial patterns of forest loss.
- Create ready-to-use datasets for machine learning models to predict deforestation.
- Share insights on Amazon forest degradation based on public and official data.

---

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

---

## 🔧 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

> ⚠️ Note: To run Earth Engine scripts, you need to add your credentials in `assets/earthengine-project.json`. However, all data analyses can be executed with the available local datasets.

---

## 📊 Data Sources

- **Legal Amazon Shapefile**: https://terrabrasilis.dpi.inpe.br/geonetwork/srv/por/catalog.search#/metadata/f2153c4a-915b-48a6-8658-963bdce7366c
- **Fire outbreaks, precipitation, and pasture data**: Extracted using Google Earth Engine (GEE) and stored in Google Cloud Storage (GCS) bucket

---

## 🚧 Roadmap

- ✅ Exploratory and spatial data analysis
- ✅ Integration of fire, population, and exploitation indicators
- 🔜 Data preparation for ML modeling
- 🔜 Forecasting Amazon deforestation using ML (Random Forest, LightGBM, etc.)
- 🔜 Interactive Dashboard and report

---

## 📬 Author

**Ricardo Fonseca**  
MBA in Data Science — Autónoma Academy  
[LinkedIn](https://www.linkedin.com/in/ricardolfonseca)