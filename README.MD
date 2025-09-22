# UrbanFlow AI: Smart Bike Demand Forecasting System

![UrbanFlow AI](Assets/Header.png)

## 🚴‍♂️ Project Overview

UrbanFlow AI is an advanced data science solution that predicts bike-sharing demand across New York City by analyzing historical Citi Bike data alongside weather patterns and traffic information. The system provides accurate 24-hour forecasts to optimize bike availability and distribution, improving operational efficiency and user experience.

## 🎯 Key Features

- **Real-time Demand Forecasting**: 24-hour bike demand predictions
- **Multi-source Data Integration**: Combines Citi Bike, weather, and traffic data
- **Interactive Dashboard**: Streamlit-based UI with multiple analysis views
- **AI-Powered Chatbot**: Natural language interface for data insights
- **Comprehensive Analytics**: Station usage maps, temporal patterns, weather impact analysis

## 📊 Data Sources

1. **Citi Bike Data**: Trip records, station information, user demographics
2. **Weather Data**: Temperature, precipitation, wind speed, seasonal patterns
3. **Traffic Data**: Vehicle density from Automatic Traffic Recorders

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML/AI**: scikit-learn, XGBoost, LightGBM, LSTM
- **NLP/Chatbot**: LangChain, HuggingFace Transformers, FAISS
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly


## 📁 Project Structure

```
UrbanFlow AI/
├── app/                    # Streamlit application
│   ├── main.py            # Main dashboard
│   └── chatbot.py         # AI chatbot implementation
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 00_project_overview.ipynb
│   ├── 01_load_explore_bike_data.ipynb
│   ├── 02_load_explore_weather_data.ipynb
│   ├── 03_load_explore_traffic_atr.ipynb
│   ├── 04-06_modeling_*.ipynb
│   ├── 08_ML_merged_data.ipynb
│   └── ml-urbanflow.ipynb
├── data/                  # Data storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned datasets
│   └── final/           # Feature-engineered datasets
├── models/               # Model utilities and trained models
├── outputs/             # Generated outputs
└── docs/               # Documentation
```

## 🔍 Key Insights

- **Weather Impact**: Precipitation reduces usage by ~40%, temperature is strongest predictor (optimal: 15-25°C)
- **Bike Type Usage**: Electric bikes used for 30% longer trips (2.5km vs 1.8km)
- **Temporal Patterns**: Weekday peaks during commute hours, weekend usage more distributed
- **Seasonal Trends**: Highest usage during summer months (June-August)


## 🤖 AI Chatbot Features

- Natural language queries about bike usage patterns
- Weather impact analysis
- Station popularity insights
- Traffic correlation explanations
- Fallback responses when API unavailable

## 🗺️ Dashboard Features

1. **Overview**: KPIs and time series visualizations
2. **Station Map**: Interactive map with usage patterns
3. **Forecasting**: ML predictions vs actual usage
4. **Exploratory Insights**: EDA plots and trend analysis
5. **AI Explainability**: SHAP/LIME model interpretations
6. **Chatbot Assistant**: Conversational interface

**Note**: This is a demonstration project showcasing data science and machine learning capabilities in urban mobility forecasting.
