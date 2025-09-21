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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/UrbanFlowAI.git
cd UrbanFlowAI
```

2. Create and activate virtual environment:
```bash
python -m venv urbanflow
source urbanflow/bin/activate  # On Windows: urbanflow\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional, for AI chatbot):
```bash
# Create .env file in project root
echo "HUGGINGFACEHUB_API_TOKEN=your_hf_token_here" > .env
```

5. Run the application:
```bash
cd app
streamlit run main.py
```

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

## 📈 Model Performance

The forecasting system employs multiple models:
- **Time Series**: ARIMA, Prophet, LSTM
- **Machine Learning**: Gradient Boosting, Random Forest, Ensemble methods
- **Evaluation Metrics**: RMSE, MAE, MAPE

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

## 🔧 Configuration

### Environment Variables

- `HUGGINGFACEHUB_API_TOKEN`: Optional HuggingFace API token for enhanced chatbot
- Without token: Basic chatbot with predefined responses

### Data Setup

Data files are expected in the following structure:
```
data/final/
├── citibike_engineered.csv
├── weather_engineered.csv
├── traffic_engineered.csv
└── merged_dataset.csv
```

## 🚀 Future Enhancements

- [ ] Real-time data integration
- [ ] Mobile app development
- [ ] API development for third-party integration
- [ ] Advanced visualization features
- [ ] Personalized recommendations
- [ ] Rebalancing optimization algorithms

## 🚀 Deployment

The application is optimized for **Streamlit Cloud** deployment:

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository and set main file to `app/main.py`
4. Deploy instantly!

See `STREAMLIT_DEPLOYMENT.md` for detailed step-by-step instructions.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- NYC Citi Bike for providing open data
- OpenWeatherMap for weather data
- NYC Department of Transportation for traffic data
- HuggingFace for AI model infrastructure

---

**Note**: This is a demonstration project showcasing data science and machine learning capabilities in urban mobility forecasting.