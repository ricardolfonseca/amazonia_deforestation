# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from config import *
from controller.controller import *
from model.Forecast_amazonia import *


def main():
    # 0) Check if the cleaned dataset exists
    if not os.path.exists(DATASET_CLEAN):
        st.error(
            f'❌ Cleaned dataset not found at:\n{DATASET_CLEAN}\n\n'
            'Please run the data preparation pipeline before opening this Report.'
        )
        st.stop()

    # 1) Introduction
    st.set_page_config(page_title='Deforestation in Legal Amazon Report', layout='wide')
    st.image(
        'assets/amazonia_forest_banner.jpg',
        use_container_width=True,
    )
    st.title('🌳 Legal Amazon Deforestation Report')
    st.markdown('---')
    st.markdown(
        '''
        ### **Welcome to the Legal Amazon Deforestation Interactive Report!**

        The Amazon Legal region covers about 59% of Brazil's territory and is home to one of the world's richest biodiversities.  
        Unfortunately, this area faces significant human pressure, especially from deforestation and wildfires, impacting climate, soil, and local communities.
        
        Because environmental sustainability is a topic I deeply care about, I chose to explore this challenge using data — diving into satellite records, climate indicators, and land use patterns to produce a comprehensive and data-driven analysis in this report. Here, you will find:
        1. **Introduction**: Context about the Amazon Legal region and Report objectives.
        2. **Interactive Map** (coming soon): Visualization of yearly loss of forest cover.
        3. **Exploratory Data Analysis (EDA)**: Charts and tables showing correlations, distributions, and historical patterns.
        4. **Machine Learning**: Predictive models to anticipate deforestation trends.

        

        > *Disclaimer: This project was developed as part of the final assessment for an MBA in Data Science, with the primary goal of applying and demonstrating the knowledge and skills acquired throughout the course. All data used was obtained from publicly available sources. While every effort was made to ensure data accuracy and integrity through proper preprocessing, integration, and validation techniques, **some inconsistencies or limitations may still be present**. Therefore, the results presented here — including predictions, trends, and visualizations — are intended for **educational and analytical purposes only**, and should not be interpreted as definitive or official figures.*

        '''
    )
    st.markdown('')

    st.subheader('🧪 About this Project')
    st.markdown('---')
    st.markdown(
    '''
    This Report is the result of my final MBA project in Data Science, focused on understanding and predicting deforestation patterns in the Brazilian Legal Amazon.
    The process involved multiple technical and analytical steps, structured around the CRISP-DM methodology, from data extraction to machine learning modeling.
    ''')

    st.markdown('#### Data Sources and Preparation')
    st.markdown(
    '''
    The core dataset is based on the DETER deforestation alerts from INPE's TerraBrasilis platform, which provides georeferenced daily records of deforestation in the Legal Amazon.
    Additional environmental and anthropogenic factors—such as wildfire hotspots, rainfall, and land use were extracted from Google Earth Engine and stored in cloud buckets for local processing.
    
    After acquiring the raw data, I performed extensive ETL (Extract, Transform, Load) procedures:
    - Filtering and cleaning shapefiles for the Legal Amazon territory.
    - Aggregating data by day, month, and municipality.
    - Integrating multiple sources into a unified time-series dataset.
        
    ''')

    st.markdown('#### Data Analysis and Feature Engineering')
    st.markdown(
    '''
    Using Python, I explored patterns and trends with visual analytics and statistical summaries.

    I then created lag features (e.g., previous day/week/month deforestation), log transformations, and normalized variables to prepare the data for machine learning.
    ''')

    st.markdown('#### Machine Learning Modeling')
    st.markdown(
    '''    
    Three rounds of regression-based models were developed to forecast short-term deforestation:
    - Baseline models: such as Lasso Regression and LightGBM with default parameters.
    - Tuned models: using pipelines and cross-validation for hyperparameter optimization.
    - Advanced setups: including early stopping, scaling, and model selection based on multiple performance metrics.

    The final models were evaluated using MAE, RMSE, and R² scores on temporal train-test splits, ensuring robustness and avoiding data leakage.
    
    *Enjoy exploring the results and insights from this project!*
            
    ''')

    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space


    # 2) Placeholder for the interactive map
    st.subheader('🔍 Interactive Map')
    st.markdown('---')
    st.markdown('**(Coming soon: this space will display the map with deforestation alerts.)**')

    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space

    # 3) Exploratory Data Analysis
    st.subheader('📊 Exploratory Data Analysis')
    st.markdown('---')

    # 3.1 Load the cleaned dataframe
    df = pd.read_csv(DATASET_CLEAN, parse_dates=['date'], index_col='date')

    # 3.2 Numeric statistics and Correlation Matrix side by side
    num_stats = df.select_dtypes(include='number').describe().round(2)
    col1, col_blank, col2 = st.columns([0.45, 0.05, 0.45])
    with col1:
        st.markdown('#### 🔢 Numeric Statistics')
        st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
        st.dataframe(num_stats, use_container_width=True)
    with col2:
        st.markdown('#### 🔗 Correlation Matrix')
        corr_method = st.selectbox(
            'Choose correlation method:',
            ['Pearson', 'Spearman', 'Kendall']
        )
        corr_mat = df.corr(method=corr_method.lower(), numeric_only=True)
        fig_corr = px.imshow(
            corr_mat,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f'Correlation Matrix ({corr_method})'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # st.markdown('#### Data Overview and Key Relationships')
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
    st.markdown(
    '''    
    The dataset used in this analysis comprises **3 165 daily observations** from the Brazilian Legal Amazon region. It includes four key variables identified as relevant drivers or indicators of deforestation:
    - **Deforestation Area (ha)**: daily area in hectares where forest cover was lost.
    - **Fires**: number of fire hotspots detected.
    - **Precipitation (mm)**: daily rainfall levels.
    - **Pasture Area (ha)**: annual pasture extent, assigned to each day to match the granularity of other variables.

    From the descriptive statistics, we observe a **highly skewed distribution** of deforestation events, with a median of 0 ha, indicating that most days record no official alerts, while some extreme cases surpass 435k ha in a single day.
    The presence of such outliers required careful normalization during data preprocessing.
    Similarly, the number of fire hotspots varies widely, ranging from 0 to 892 per day, with a mean of 82.65, reinforcing the episodic and spatially concentrated nature of wildfires in the region.
    
    The correlation matrix reveals some insights into the interdependence between environmental factors:
    - A **moderate positive correlatio**n is observed between the **number of fires and deforestation**, suggesting that fire activity often accompanies forest loss.
    - Precipitation shows a **moderate negative correlation** with fires and a weaker inverse relationship with deforestation, indicating that drier periods are more conducive to fire propagation and possibly to forest clearing operations.
    ''')
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space

    

    # 3.3 Annual charts for each main variable (two per row)
    st.markdown('### 📈 Annual Trends')
    df_yearly = df.resample('YE').sum()
    metrics = {
        'deforestation_area_ha': 'Deforestation area (ha)',
        'fires': 'Fires',
        'precipitation_mm': 'Precipitation (mm)',
        'pasture_area_ha': 'Pasture area (ha/day)'
    }
    metric_items = list(metrics.items())
    for i in range(0, len(metric_items), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(metric_items):
                col_key, label = metric_items[i + j]
                fig_year = px.line(
                    x=df_yearly.index.year,
                    y=df_yearly[col_key],
                    labels={'x': 'Year', 'y': label},
                    markers=True,
                    title=f'Yearly Total: {label}'
                )
                cols[j].plotly_chart(fig_year, use_container_width=True)


    # st.markdown('#### Distribution of variables - Histograms')
    st.markdown(
    '''
    The annual aggregates displayed in the Report provide important temporal context of deforestation in the Legal Amazon.

    1. **Deforestation Area (ha/year)**
    Between 2016 and 2024, deforestation shows strong variability. A sharp increase is observed in 2020, likely related to political and enforcement changes during that period. After a drop in 2021, deforestation rose again, peaking in 2024. The drop in 2025 is partial and reflects incomplete data collection for that year.

    2. **Fire Incidence (hotspots/year)**
    The number of fire hotspots follows a similar yet noisier pattern, peaking in 2019 and 2024, closely matching deforestation surges. This alignment reinforces the hypothesis of a structural link between fire and forest clearing, as previously suggested in the correlation analysis.

    3. **Precipitation (mm/year)**
    Rainfall exhibits less interannual volatility, but the lowest precipitation values in 2016 and 2025 coincide with above-average deforestation years. Notably, the dry year of 2024, which recorded reduced rainfall compared to previous years, aligns with the peak in both fires and forest loss. This pattern supports the negative correlation between precipitation and fire incidence observed earlier.

    4. **Pasture Area (ha/day)**
    Pasture area presents a steady upward trend from 2016 to 2024, reflecting continued land conversion for livestock and agriculture. Its' cumulative growth over time may help explain the structural pressures on forest cover. The drop in 2025 is due to data unavailability rather than an actual reduction in pasture area.
    ''')    
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space

    # 3.4 Histograms (two per row)
    st.markdown('### 📊 Distribution of variables (Histograms)')
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for i in range(0, len(num_cols), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(num_cols):
                col_name = num_cols[i + j]
                fig_hist = px.histogram(
                    df,
                    x=col_name,
                    nbins=30,
                    # marginal='box',
                    title=f'Histogram of {col_name}'
                )
                cols[j].plotly_chart(fig_hist, use_container_width=True)

    # st.markdown('#### Variable Distributions – Insights for Modeling')
    st.markdown(
    '''
    Understanding the distribution of each variable is a crucial step in preparing data for machine learning. It allows us to detect skewness, outliers, and scaling issues that can negatively affect model performance if not addressed.

    1. **Deforestation Area (ha/year)**
    The distribution is **highly right-skewed**, with the vast majority of observations clustered at or near **zero hectares**. A small number of extreme events represent very high values, which are likely tied to cumulative or delayed reporting in the DETER alert system. These extreme values have **high influence on the mean**.
    
    2. **Fire Incidence (hotspots/year)**
    Similarly, the number of daily fire hotspots follows a **long-tailed distribution**. 
    
    3. **Precipitation (mm/year)**
    Unlike the other variables, precipitation exhibits a more **symmetrical, slightly left-skewed** distribution.
    
    4. **Pasture Area (ha/day)**
    The histogram reveals an **almost discrete distribution**, reflecting the **yearly update frequency** of this variable, where each value corresponds to a year.  
    ''')
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space


    # 3.5 Boxplots (two per row)
    st.markdown('### 📦 Boxplots')
    for i in range(0, len(num_cols), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(num_cols):
                col_name = num_cols[i + j]
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=df[col_name], name=col_name, boxmean=True))
                fig_box.update_layout(
                    title=f'Boxplot of {col_name}',
                    yaxis_title=col_name,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=300
                )
                cols[j].plotly_chart(fig_box, use_container_width=True)
    
    # st.markdown('#### Boxplots – Outlier Detection and Variable Spread')
    st.markdown(
    '''
    Boxplots offer a compact visual summary of each variable's distribution, highlighting central tendency, dispersion, and the presence of outliers.

    1. **Deforestation Area (ha/year)**
    This variable displays an **extreme number of outliers**, far above the upper quartile. While the median is near zero (confirming that most days register no deforestation alerts), there are a significant number of high-impact events. 

    2. **Fire Incidence (hotspots/year)**
    The distribution of fire hotspots also contains **many high-end outliers**, especially above the 75th percentile. The central box is compressed, indicating that **most values are tightly concentrated** below ~100.  

    3. **Precipitation (mm/year)**
    Precipitation shows a **more symmetrical boxplot**, with a wide interquartile range. Outliers are present but not as extreme as in the previous variables.

    4. **Pasture Area (ha/day)**
    This boxplot reflects the **annual update cycle** of this variable: the values are concentrated in **batches**, forming a stepwise pattern. There are **no visible outliers**, and the interquartile range is narrow.
    ''')
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space


     # 4) Machine Learning
    st.subheader('🤖 Machine Learning')
    st.markdown('---')

    # st.markdown('#### Machine Learning: Forecasting Deforestation Trends')
    st.markdown(
    '''
    The main goal of this project is to **anticipate deforestation in the Brazilian Legal Amazon**, enabling earlier interventions and more effective policymaking. To achieve this, we developed predictive models capable of estimating short-term forest loss based on historical patterns and environmental drivers.

    However, before modeling, it was crucial to **prepare the data efficiently**. This involved:

    - **Transforming skewed variables** (e.g., deforestation area) using log transformation to stabilize variance;  
    - **Normalizing the features** to ensure comparability across different scales;  
    - **Engineering lag features** and rolling averages to capture temporal dependencies;  
    - **Managing outliers** that could bias learning, especially in fire and deforestation variables.

    With the cleaned and enriched dataset, we implemented and compared three different machine learning models:
    ''')




    # 4.1) Show three “cards” explaining each model, using the same colors as the legend
    col_lgbm, col_lasso, col_mlp = st.columns(3)

    # LightGBM card (blue border/text)
    with col_lgbm:
        st.markdown(
            '''
            <div style='
                border: 2px solid #1f77b4;
                border-radius: 12px;
                padding: 10px;
                text-align: center;
                margin-bottom: 8px;
            '>
                <h4 style='margin: 0; color: #1f77b4;'>LightGBM</h4>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            - Uses gradient boosting on decision trees  
            - Great for large datasets and capturing non-linear relationships  
            '''
        )

    # Lasso card (orange border/text)
    with col_lasso:
        st.markdown(
            '''
            <div style='
                border: 2px solid #ff7f0e;
                border-radius: 12px;
                padding: 10px;
                text-align: center;
                margin-bottom: 8px;
            '>
                <h4 style='margin: 0; color: #ff7f0e;'>Lasso</h4>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            - Linear regression with L₁ regularization  
            - Good for feature selection and preventing overfitting  
            '''
        )

    # MLP card (red border/text)
    with col_mlp:
        st.markdown(
            '''
            <div style='
                border: 2px solid #d62728;
                border-radius: 12px;
                padding: 10px;
                text-align: center;
                margin-bottom: 8px;
            '>
                <h4 style='margin: 0; color: #d62728;'>MLP</h4>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            - A feed-forward neural network with hidden layers  
            - Useful for capturing complex, non-linear patterns in the data  
            '''
        )

    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
    st.markdown(
        '''
        To ensure a rigorous evaluation, we designed the modeling pipeline around **three rounds of testing**, each with increasing levels of sophistication:

        1. **Round 1 - Baseline models**  
        Initial models with minimal preprocessing and default parameters, to establish reference performance.

        2. **Round 2 - Hyperparameter tuning**  
        Models optimized via grid search and cross-validation, with normalized input data.

        3. **Round 3 - Advanced techniques**  
        Introduction of pipelines, early stopping, and additional validation steps to improve generalization and prevent overfitting.

        This structured approach allowed us not only to identify the best-performing model, but also to understand **how data preparation and model configuration affect forecasting quality**, ensuring both robustness and transparency in the modeling process.
        ''')
    st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space


    # 4.2) Initialize session_state flags / objects if missing
    if 'forecast_obj' not in st.session_state:
        st.session_state.forecast_obj = None
    if 'round1_done' not in st.session_state:
        st.session_state.round1_done = False
    if 'round2_done' not in st.session_state:
        st.session_state.round2_done = False
    if 'round3_done' not in st.session_state:
        st.session_state.round3_done = False

    
    # === Round 1 (Always visible) ===
    #
    st.markdown('### 🔹 Round 1 (Baseline)')
    st.markdown(
        '''
        In Round 1, we train LightGBM, Lasso, and MLP on raw (uncleaned) features with baseline hyperparameters.
        '''
    )

    def run_round1():
        forecast = Forecast_amazonia(test_size=0.2)
        st.session_state.forecast_obj = forecast
        forecast.train_test_round_one()
        # Store the Round 1 metrics‐dict
        st.session_state.results_round1 = forecast._Forecast_amazonia__round_results.get('Round 1', {})
        st.session_state.round1_done = True

    if not st.session_state.round1_done:
        st.button('Run Round 1', key='btn_run_r1', on_click=run_round1)
    else:
        # 1) Metrics plot and table side by side
        col1, col2 = st.columns(2)
        with col1:
            plot_metrics_plotly(
                st.session_state.results_round1,
                title='Round 1: MAE / RMSE / R² (All Models)'
            )
        with col2:
            # HTML spacer: 2 line-breaks worth of vertical space
            st.markdown('<br><br><br>', unsafe_allow_html=True)
            df1 = pd.DataFrame(st.session_state.results_round1).T.rename_axis('Model')
            st.dataframe(df1.round(4), use_container_width=True)

        # 2) Monthly Actual vs Predicted plot (full width)
        plot_monthly_deforestation('Round 1')

        st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
        st.markdown(
            '''
            LightGBM made the closest predictions (error ~3.4) and explained about 39% of the variation, beating Lasso and MLP.
            
            Lasso was okay (error ~3.5, explained ~37%), and MLP struggled the most (error ~5.5, negative R²), showing that simple tree-based models work better on these raw features.
            ''')
    #
    # === Round 2 (only after Round 1 completes) ===
    #
    if st.session_state.round1_done:
        st.markdown('---')
        st.markdown('### 🔹 Round 2 (Tuned + Normalized)')
        st.markdown(
            '''
            In Round 2, we hyperparameter‐tune each algorithm and normalize features before training.
            '''
        )

        def run_round2():
            forecast = st.session_state.forecast_obj
            forecast.train_test_round_two()
            st.session_state.results_round2 = forecast._Forecast_amazonia__round_results.get('Round 2', {})
            st.session_state.round2_done = True

        if not st.session_state.round2_done:
            st.button('Run Round 2', key='btn_run_r2', on_click=run_round2)
        else:
            # 1) Metrics plot and table side by side
            col3, col4 = st.columns(2)
            with col3:
                plot_metrics_plotly(
                    st.session_state.results_round2,
                    title='Round 2: MAE / RMSE / R² (All Models)'
                )
            with col4:
                # HTML spacer: 2 line-breaks worth of vertical space
                st.markdown('<br><br><br>', unsafe_allow_html=True)
                df2 = pd.DataFrame(st.session_state.results_round2).T.rename_axis('Model')
                st.dataframe(df2.round(4), use_container_width=True)

            # 2) Monthly Actual vs Predicted plot (full width)
            plot_monthly_deforestation('Round 2')

            st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
            st.markdown(
                '''
                After tuning and scaling, LightGBM's RMSE rose slightly to ~3.58 (R² ≈ 0.33), and Lasso stayed about the same (RMSE ≈ 3.51, R² ≈ 0.36).
                
                The MLP produced wildly exaggerated predictions—spiking to tens of millions of hectares in some months—which shows it is still unstable (RMSE ≈ 4.91, negative R²).
                
                In short, none of the models improved over Round 1, and the MLP's huge outliers underscore that LightGBM remains the most reliable choice.
                ''')

    #
    # === Round 3 (only after Round 2 completes) ===
    #
    if st.session_state.round2_done:
        st.markdown('---')
        st.markdown('### 🔹 Round 3 (Advanced Methods)')
        st.markdown(
            '''
            In Round 3, we use early stopping (LightGBM), LassoCV with TimeSeriesSplit, and an MLP pipeline  
            with scaling + early stopping.
            '''
        )

        def run_round3():
            forecast = st.session_state.forecast_obj
            forecast.train_test_round_three()
            st.session_state.results_round3 = forecast._Forecast_amazonia__round_results.get('Round 3', {})
            st.session_state.round3_done = True

        if not st.session_state.round3_done:
            st.button('Run Round 3', key='btn_run_r3', on_click=run_round3)
        else:
            # 1) Metrics plot and table side by side
            col5, col6 = st.columns(2)
            with col5:
                plot_metrics_plotly(
                    st.session_state.results_round3,
                    title='Round 3: MAE / RMSE / R² (All Models)'
                )
            with col6:
                # HTML spacer: 2 line-breaks worth of vertical space
                st.markdown('<br><br><br>', unsafe_allow_html=True)
                df3 = pd.DataFrame(st.session_state.results_round3).T.rename_axis('Model')
                st.dataframe(df3.round(4), use_container_width=True)

            # 2) Monthly Actual vs Predicted plot (full width)
            plot_monthly_deforestation('Round 3')

            st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space
            st.markdown(
                '''
                With early stopping and cross-validation, LightGBM ES again wins (RMSE ≈ 3.36, R² ≈ 0.41), slightly ahead of the MLP pipeline (RMSE ≈ 3.48, R² ≈ 0.37) and LassoCV (RMSE ≈ 3.49, R² ≈ 0.37).
                
                The MLP pipeline has improved dramatically—no extreme spikes—but LightGBM ES remains the most accurate and reliable choice.
                ''')


# === Final Recommendation (only after Round 3 completes) ===
#
    if st.session_state.round3_done:
        st.markdown('---')
        # Show a button; only when the user clicks does the recommendation appear
        if st.button('🎯 Show Final Recommendation', key='btn_final_rec'):
            st.markdown('## 🎯 Final Recommendation and Conclusions')

            st.markdown(
                '''
                After comparing MAE, RMSE, and R² across all three model families and all three rounds, the clear winner is:
                ''')
            
            st.markdown(
                '''
                <div style='
                    border: 2px solid #1f77b4;
                    border-radius: 12px;
                    padding: 15px;
                    max-width: 420px;
                    margin: 5px 0px;
                '>
                <h4 style='margin: 0 0 8px 0; color: #1f77b4; text-align: center;'>
                    🏆 LightGBM with early stopping
                </h4>
                <ul style='margin-top: 4px; margin-left: 20px;'>
                    <li>MAE = 2.6350</li>
                    <li>RMSE = 3.3602 ← lowest of all 9 configurations</li>
                    <li>R² = 0.4147 ← highest R² among the Round 3 models</li>
                </ul>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            st.markdown('<br>', unsafe_allow_html=True)             # HTML spacer: line-breaks worth of vertical space

            st.markdown(
                '''
                **LightGBM with early stopping** achieved the smallest RMSE on the hold-out set (3.3602), while also maintaining the best R² (0.4147).

                Because the goal is to minimize RMSE (and maximize R²), LightGBM (Advanced, Round 3) is the recommended model:

                1. **Best Accuracy**  
                - Its RMSE of 3.3602 is the lowest among all model+round combinations.
                2. **Strong Explanatory Power**  
                - R² of 0.4147 means it explains about 41% of the variance on unseen data.
                3. **Speed & Flexibility**  
                - LightGBM trains relatively quickly and handles non-linear patterns well.  
                - Early stopping in Round 3 also prevents overfitting, making the model more robust.
                
                **In conclusion, LightGBM with early stopping** produces the lowest RMSE and highest R², making it the best all-around pick for forecasting Amazon deforestation under this pipeline.  
                '''
            )
        
            # st.markdown('---')
            st.markdown(
            '''
            ---
            I hope you enjoyed this report as much as I enjoyed creating it!
            If you have any questions, feedback, or suggestions for improvement, please feel free to reach out.
            
            

            >🌱*If we want to change the future of the forest, we must first understand its present — and data is one of the strongest tools we have to do so.*

            ''')

if __name__ == '__main__':
    main()