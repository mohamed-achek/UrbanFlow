import streamlit as st
from chatbot import get_chatbot_response
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

# Import model utilities
try:
    from model_utils import load_model_for_streamlit, predict_for_streamlit, get_model_info_for_streamlit
    MODEL_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Model utilities not available: {e}")
    MODEL_AVAILABLE = False

st.set_page_config(
    page_title="UrbanFlow AI: Smart Bike Demand Forecasting in NYC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header section with logo - update styles to make image smaller
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .logo-img {
        max-width: 50%; /* Reduced from 70% to 50% */
        height: auto;
    }
    .title-text {
        color: #1E88E5;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0 !important;
    }
    .subtitle-text {
        font-size: 1.2rem !important;
        color: #666;
        margin-top: 0 !important;
    }
    /* Added styles to center the image */
    .stImage {
        text-align: center;
        margin: 0 auto;
        display: block;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Display header image - fix the image path issue
# Try multiple possible paths for the image
possible_paths = [
    "Assets/Header.png",     # From root directory (when run with streamlit)
    "../Assets/Header.png",  # From app directory (fallback)
    "./Assets/Header.png",   # Current directory relative
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "Assets", "Header.png")  # Absolute path
]

# Check if any of the possible paths exist
header_image_path = None
for path in possible_paths:
    if os.path.exists(path):
        header_image_path = path
        break

# If image not found, try to create the assets directory and display a message
if header_image_path is None:
    st.warning("🖼️ Header image not found. Looking for 'Header.png' in Assets directory.")
    st.info(f"📁 Current working directory: {os.getcwd()}")
    st.info(f"📍 Script location: {os.path.dirname(__file__)}")
    
    # Create a styled title as fallback
    st.markdown('<h1 class="title-text">UrbanFlow AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Smart Bike Demand Forecasting in NYC</p>', unsafe_allow_html=True)
else:
    # Image found, display it with responsive sizing
    col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns for layout
    with col2:  # Use the middle column to display the image
        try:
            image = Image.open(header_image_path)
            st.image(image, width=500, caption="UrbanFlow AI - Smart Urban Mobility Analytics")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.markdown('<h1 class="title-text">UrbanFlow AI</h1>', unsafe_allow_html=True)
            st.markdown('<p class="subtitle-text">Smart Bike Demand Forecasting in NYC</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("UrbanFlow AI Controls")
st.sidebar.markdown("Select filters and options to explore bike demand in NYC.")

# Interactive controls that affect the data
selected_stations = st.sidebar.multiselect(
    "Select Station(s)", 
    options=["All", "Central Park South", "Times Square", "Brooklyn Bridge", "Wall Street"], 
    default=["All"]
)

# Date range filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=[datetime(2023, 1, 1).date(), datetime(2023, 12, 31).date()],
    min_value=datetime(2023, 1, 1).date(),
    max_value=datetime(2023, 12, 31).date()
)

# Weather filters
temp_range = st.sidebar.slider("Temperature (°C)", min_value=-10, max_value=40, value=(0, 30))
precip_range = st.sidebar.slider("Precipitation (mm)", min_value=0, max_value=50, value=(0, 10))
wind_range = st.sidebar.slider("Wind Speed (km/h)", min_value=0, max_value=50, value=(0, 20))

# Additional options
include_traffic = st.sidebar.checkbox("Include Traffic Data", value=False)
model_selector = st.sidebar.selectbox("Model Selector", options=["Random Forest", "LSTM"], index=0)

# Apply filters button
if st.sidebar.button("Apply Filters") or True:  # Auto-apply filters on any change
    pass

# Download button
#if st.sidebar.button("Download Filtered Data"):
#    st.sidebar.success("Data download feature coming soon!")

# Show current filter summary in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Filter Summary")
if len(date_range) == 2:
    st.sidebar.write(f"📅 **Date**: {date_range[0]} to {date_range[1]}")
st.sidebar.write(f"🌡️ **Temperature**: {temp_range[0]}°C to {temp_range[1]}°C")
st.sidebar.write(f"🌧️ **Precipitation**: {precip_range[0]}-{precip_range[1]} mm")
st.sidebar.write(f"💨 **Wind**: {wind_range[0]}-{wind_range[1]} km/h")
st.sidebar.write(f"🚉 **Stations**: {', '.join(selected_stations) if 'All' not in selected_stations else 'All stations'}")
st.sidebar.write(f"🤖 **Model**: {model_selector}")

# Model status indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Model Status")
if MODEL_AVAILABLE:
    if st.session_state.get('model_loaded', False):
        st.sidebar.success("✅ Trained model loaded")
        model_info = get_model_info_for_streamlit()
        if model_info.get('metadata'):
            st.sidebar.write(f"📊 R² Score: {model_info['metadata'].get('r2_score', 'N/A'):.3f}")
    else:
        st.sidebar.warning("⚠️ Model available but not loaded")
else:
    st.sidebar.error("❌ Model utilities not available")
    st.sidebar.error("🚫 Real predictions unavailable")

# Main page tabs
st.title("UrbanFlow AI: Smart Bike Demand Forecasting in NYC")

# Data notice
st.info("""
🗂️ **Data Notice:** Using real NYC bike share data and weather information.
- Historical bike demand patterns from NYC CitiBike stations
- Weather data from OpenWeather API for accurate forecasting
- Traffic data from NYC Department of Transportation
""")

# Helper function to load and filter sample data
@st.cache_data
def load_sample_data():
    """Load real NYC bike demand and weather data"""
    # Try to load sample data first
    sample_paths = [
        "data/samples/merged_dataset_sample.csv",
        "data/samples/citibike_sample.csv",
        "data/final/merged_dataset_20250720_170853.csv",  # Local fallback
        "data/final/citibike_engineered_20250719_161523.csv"
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    if "sample" in path:
                        pass
                    else:
                        st.info(f"✅ Loaded full local dataset: {len(df):,} rows (limited to 10,000 for performance)")
                        df = df.head(10000)  # Limit for performance
                    return df
            except Exception as e:
                continue
    
    # If no datasets found, return None instead of fake data
    st.error("❌ No datasets found. Please ensure data files are available in the data/ directory.")
    st.info("📁 Expected files: data/final/*.csv or data/processed/*.csv")
    return None

def filter_data(df, stations, temp_range, precip_range, wind_range, date_range):
    """Filter data based on sidebar controls"""
    filtered_df = df.copy()
    
    # Filter by stations
    if "All" not in stations and len(stations) > 0:
        if 'start_station_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['start_station_name'].isin(stations)]
        elif 'start_station_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['start_station_id'].isin(stations)]
    
    # Filter by temperature
    if 'temperature_2m' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['temperature_2m'] >= temp_range[0]) & 
            (filtered_df['temperature_2m'] <= temp_range[1])
        ]
    
    # Filter by precipitation
    if 'precipitation' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['precipitation'] >= precip_range[0]) & 
            (filtered_df['precipitation'] <= precip_range[1])
        ]
    
    # Filter by wind speed
    if 'wind_speed_10m' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['wind_speed_10m'] >= wind_range[0]) & 
            (filtered_df['wind_speed_10m'] <= wind_range[1])
        ]
    
    # Filter by date range
    if 'datetime' in filtered_df.columns and len(date_range) == 2:
        filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['datetime'].dt.date >= start_date.date()) & 
            (filtered_df['datetime'].dt.date <= end_date.date())
        ]
    
    return filtered_df

# Load data once
raw_data = load_sample_data()

# Check if data was loaded successfully
if raw_data is None:
    st.error("❌ Cannot continue without data. Please check your data directory.")
    st.stop()

# Apply filters to the data
sample_data = filter_data(raw_data, selected_stations, temp_range, precip_range, wind_range, date_range)
tabs = st.tabs([
    "Dashboard Overview",
    "Forecasting Panel",
    "Exploratory Insights",
    "AI & Explainability",
    "Chatbot Assistant"
])

with tabs[0]:
    st.header("Dashboard Overview")
    
    # Show active filters
    st.subheader("🔍 Active Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stations", f"{len(selected_stations) if 'All' not in selected_stations else 'All'}")
        st.metric("Temperature Range", f"{temp_range[0]}°C to {temp_range[1]}°C")
    with col2:
        st.metric("Precipitation", f"{precip_range[0]}-{precip_range[1]} mm")
        st.metric("Wind Speed", f"{wind_range[0]}-{wind_range[1]} km/h")
    with col3:
        if len(date_range) == 2:
            st.metric("Date Range", f"{date_range[0]} to {date_range[1]}")
        st.metric("Data Points", f"{len(sample_data):,}")
    
    # Key Performance Indicators
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trips = sample_data['demand_count'].sum() if 'demand_count' in sample_data.columns and not sample_data.empty else 0
        st.metric("Total Trips", f"{total_trips:,}")
    
    with col2:
        avg_temp = sample_data['temperature_2m'].mean() if 'temperature_2m' in sample_data.columns and not sample_data.empty else 0
        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    with col3:
        if 'hour' in sample_data.columns and not sample_data.empty:
            peak_hour = sample_data.groupby('hour')['demand_count'].sum().idxmax()
        else:
            peak_hour = 17
        st.metric("Peak Hour", f"{peak_hour}:00")
    
    with col4:
        if 'start_station_name' in sample_data.columns and not sample_data.empty:
            busiest_station = sample_data.groupby('start_station_name')['demand_count'].sum().idxmax()
        else:
            busiest_station = "No data"
        st.metric("Busiest Station", busiest_station[:15] + "..." if len(busiest_station) > 15 else busiest_station)
    
    # Check if we have data after filtering
    if sample_data.empty:
        st.warning("⚠️ No data matches the current filter criteria. Please adjust your filters.")
    else:
        # Time series chart
        st.subheader("📈 Trip Count Over Time")
        if 'datetime_hour' in sample_data.columns and 'demand_count' in sample_data.columns:
            # Aggregate by day for better visualization
            daily_data = sample_data.copy()
            daily_data['date'] = pd.to_datetime(daily_data['datetime_hour']).dt.date
            daily_summary = daily_data.groupby('date')['demand_count'].sum().reset_index()
            
            fig = px.line(daily_summary, x='date', y='demand_count', 
                         title='Bike Trip Demand Over Time (Filtered Data)',
                         labels={'date': 'Date', 'demand_count': 'Total Daily Trips'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time series data not available in current dataset.")
        
        # Weather correlation
        st.subheader("🌤️ Weather Impact on Bike Usage")
        if 'temperature_2m' in sample_data.columns and 'demand_count' in sample_data.columns:
            fig = px.scatter(sample_data, x='temperature_2m', y='demand_count', 
                            title='Temperature vs Trip Count (Filtered Data)',
                            labels={'temperature_2m': 'Temperature (°C)', 'demand_count': 'Number of Trips'},
                            color='precipitation' if 'precipitation' in sample_data.columns else None,
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Weather correlation data not available.")
        
        # Show filter impact
        original_count = len(raw_data)
        filtered_count = len(sample_data)
        filter_percentage = (filtered_count / original_count) * 100 if original_count > 0 else 0
        
        st.info(f"📊 Filters applied: Showing {filtered_count:,} of {original_count:,} data points ({filter_percentage:.1f}%)")

with tabs[1]:
    st.header("Forecasting Panel")
    
    # Load model if available
    if MODEL_AVAILABLE:
        with st.spinner("Loading trained model..."):
            if 'model_loaded' not in st.session_state:
                st.session_state.model_loaded = load_model_for_streamlit("random_forest")
            
            if st.session_state.model_loaded:
                model_info = get_model_info_for_streamlit()
                st.success(f"✅ Using trained {model_info.get('model_type', 'Unknown')} model")
                
                # Show model performance
                if model_info.get('metadata'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{model_info['metadata'].get('r2_score', 0):.3f}")
                    with col2:
                        st.metric("RMSE", f"{model_info['metadata'].get('rmse', 0):.2f}")
                    with col3:
                        st.metric("MAE", f"{model_info['metadata'].get('mae', 0):.2f}")
    
    # Forecasting section
    st.subheader("🔮 Real-Time Demand Prediction")
    
    # Input parameters for prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather Conditions")
        pred_temp = st.slider("Temperature (°C)", -10, 40, 20)
        pred_precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
        pred_wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)
    
    with col2:
        st.subheader("Time & Context")
        pred_hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
        pred_is_weekend = st.checkbox("Weekend", value=datetime.now().weekday() >= 5)
        pred_is_peak = st.checkbox("Peak Hour (7-9 AM, 5-7 PM)", 
                                  value=(7 <= pred_hour <= 9) or (17 <= pred_hour <= 19))
    
    # Make prediction
    if st.button("🚀 Predict Demand", type="primary"):
        if MODEL_AVAILABLE and st.session_state.get('model_loaded', False):
            try:
                # Prepare input data
                input_data = {
                    'temperature': pred_temp,
                    'precipitation': pred_precip,
                    'wind_speed': pred_wind,
                    'hour': pred_hour,
                    'is_weekend': pred_is_weekend,
                    'is_peak_hour': pred_is_peak
                }
                
                # Make prediction
                prediction, details = predict_for_streamlit(input_data)
                
                # Display results
                st.success(f"🎯 **Predicted Demand: {prediction:.0f} trips**")
                
                # Show prediction details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Details")
                    st.write(f"**Confidence**: {details['model_confidence']:.1%}")
                    st.write(f"**Model**: {model_selector}")
                    
                    # Show feature contributions
                    if details.get('feature_importance'):
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame([
                            {"Feature": k, "Importance": v} 
                            for k, v in details['feature_importance'].items()
                        ]).sort_values('Importance', ascending=False)
                        
                        fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                                       orientation='h', title='Feature Importance')
                        st.plotly_chart(fig_imp, use_container_width=True)
                
                with col2:
                    st.subheader("Input Summary")
                    input_summary = pd.DataFrame([
                        {"Parameter": "Temperature", "Value": f"{pred_temp}°C"},
                        {"Parameter": "Precipitation", "Value": f"{pred_precip}mm"},
                        {"Parameter": "Wind Speed", "Value": f"{pred_wind}km/h"},
                        {"Parameter": "Hour", "Value": f"{pred_hour}:00"},
                        {"Parameter": "Weekend", "Value": "Yes" if pred_is_weekend else "No"},
                        {"Parameter": "Peak Hour", "Value": "Yes" if pred_is_peak else "No"},
                    ])
                    st.dataframe(input_summary, hide_index=True)
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.warning("⚠️ Trained model not available. Using sample prediction.")
            # Fallback prediction
            base_prediction = 50
            if pred_is_peak:
                base_prediction *= 1.5
            if pred_is_weekend:
                base_prediction *= 0.8
            if pred_temp < 0 or pred_temp > 35:
                base_prediction *= 0.7
            if pred_precip > 5:
                base_prediction *= 0.5
            
            st.info(f"📊 Sample Prediction: {base_prediction:.0f} trips")
    
    # 24-Hour Forecast
    st.subheader("📈 24-Hour Demand Forecast")
    
    @st.cache_data
    def generate_model_forecast(model_name, base_temp=20, base_precip=0, base_wind=10):
        if not (MODEL_AVAILABLE and st.session_state.get('model_loaded', False)):
            st.error("❌ Model not available. Cannot generate predictions.")
            st.info("💡 Please ensure the trained model is loaded to see real predictions.")
            return None
        
        current_time = datetime.now()
        future_hours = [current_time + timedelta(hours=i) for i in range(24)]
        forecasted_trips = []
        
        for hour_dt in future_hours:
            hour = hour_dt.hour
            is_weekend = hour_dt.weekday() >= 5
            is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
            
            try:
                input_data = {
                    'temperature_2m': base_temp,
                    'precipitation': base_precip,
                    'wind_speed_10m': base_wind,
                    'hour': hour,
                    'is_weekend': 1 if is_weekend else 0,
                    'is_peak_hour': 1 if is_peak else 0
                }
                prediction, _ = predict_for_streamlit(input_data)
                forecasted_trips.append(prediction)
            except Exception as e:
                st.error(f"❌ Prediction failed for hour {hour}: {str(e)}")
                return None
        
        return pd.DataFrame({
            'datetime': future_hours,
            'forecasted': forecasted_trips
        })
    
    # Display model info
    if MODEL_AVAILABLE and st.session_state.get('model_loaded', False):
        st.success(f"🤖 **Using Trained Model**: {model_selector} - Real predictions from Random Forest")
    else:
        st.warning("⚠️ **Model Not Available** - Please load the trained model to generate predictions")
        st.stop()
    
    forecast_data = generate_model_forecast(model_selector, pred_temp, pred_precip, pred_wind)
    
    if forecast_data is None:
        st.stop()
    
    # Plotting forecast vs actual
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_data['datetime'],
        y=forecast_data['forecasted'],
        mode='lines+markers',
        name='Forecasted',
        line=dict(color='blue')
    ))
    
    # Add actual data (only for past hours)
    if 'actual' in forecast_data.columns:
        actual_data = forecast_data.dropna(subset=['actual'])
        if not actual_data.empty:
            fig.add_trace(go.Scatter(
                x=actual_data['datetime'],
                y=actual_data['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='red')
            ))
    
    fig.update_layout(
        title='24-Hour Bike Demand Forecast',
        xaxis_title='Time',
        yaxis_title='Number of Trips',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

with tabs[2]:
    st.header("Exploratory Insights")
    
    if sample_data.empty:
        st.warning("⚠️ No data available for the current filters. Please adjust your criteria.")
    else:
        # Usage patterns by hour
        st.subheader("📊 Usage Patterns by Hour of Day (Filtered Data)")
        if 'hour' in sample_data.columns and 'demand_count' in sample_data.columns:
            hourly_usage = sample_data.groupby('hour')['demand_count'].mean().reset_index()
            fig = px.bar(hourly_usage, x='hour', y='demand_count',
                        title=f'Average Trips by Hour of Day ({len(sample_data):,} data points)',
                        labels={'hour': 'Hour of Day', 'demand_count': 'Average Trips'},
                        color='demand_count',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Sample hourly pattern
            hours = list(range(24))
            trips = [15, 10, 8, 12, 25, 45, 85, 95, 75, 55, 60, 65, 70, 75, 80, 75, 85, 95, 90, 70, 50, 35, 25, 20]
            hourly_df = pd.DataFrame({'hour': hours, 'avg_trips': trips})
            fig = px.bar(hourly_df, x='hour', y='avg_trips',
                        title='Average Trips by Hour of Day (Sample Data)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show weather impact based on current data
        st.subheader("🌤️ Weather Impact Analysis (Current Filters)")
        if 'temperature' in sample_data.columns and 'precipitation' in sample_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Temperature distribution
                fig_temp = px.histogram(sample_data, x='temperature', nbins=20,
                                       title='Temperature Distribution in Filtered Data',
                                       labels={'temperature': 'Temperature (°C)', 'count': 'Frequency'})
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                # Precipitation vs trips
                fig_precip = px.scatter(sample_data, x='precipitation', y='demand_count',
                                       title='Precipitation vs Trip Count',
                                       labels={'precipitation': 'Precipitation (mm)', 'demand_count': 'Trips'},
                                       color='temperature_2m')
                st.plotly_chart(fig_precip, use_container_width=True)
        
        # Station-specific insights
        if 'start_station_id' in sample_data.columns:
            st.subheader("🚉 Station-Specific Insights")
            station_summary = sample_data.groupby('start_station_id').agg({
                'demand_count': ['sum', 'mean', 'std'],
                'temperature_2m': 'mean',
                'precipitation': 'mean'
            }).round(2)
            
            station_summary.columns = ['Total Trips', 'Avg Trips', 'Trip Std Dev', 'Avg Temp (°C)', 'Avg Precip (mm)']
            st.dataframe(station_summary, use_container_width=True)
        
        # Summary insights based on filtered data
        st.subheader("📋 Key Insights from Filtered Data")
        total_trips = sample_data['demand_count'].sum()
        avg_temp = sample_data['temperature_2m'].mean() if 'temperature_2m' in sample_data.columns else None
        avg_precip = sample_data['precipitation'].mean() if 'precipitation' in sample_data.columns else None
        
        insights = []
        insights.append(f"• Total trips in filtered period: {total_trips:,}")
        if avg_temp is not None:
            insights.append(f"• Average temperature: {avg_temp:.1f}°C")
        if avg_precip is not None:
            insights.append(f"• Average precipitation: {avg_precip:.1f}mm")
        if 'hour' in sample_data.columns:
            peak_hour = sample_data.groupby('hour')['demand_count'].sum().idxmax()
            insights.append(f"• Peak usage hour: {peak_hour}:00")
        
        for insight in insights:
            st.write(insight)

with tabs[3]:
    st.header("AI & Explainability")
    
    st.subheader("Model Feature Importance")
    
    # Sample feature importance data
    features = pd.DataFrame({
        'feature': ['Temperature', 'Hour of Day', 'Weekday/Weekend', 'Precipitation', 
                   'Wind Speed', 'Station Location', 'Season', 'Holiday'],
        'importance': [0.25, 0.20, 0.15, 0.12, 0.08, 0.10, 0.07, 0.03]
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(features, x='importance', y='feature', orientation='h',
                title='Feature Importance in Bike Demand Prediction',
                labels={'importance': 'Importance Score', 'feature': 'Features'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Model explanation
    st.subheader("Model Explanation")
    st.write("""
    **Key Findings from Model Analysis:**
    
    • **Temperature** is the strongest predictor, with optimal usage at 15-25°C
    • **Time of day** significantly affects demand, with clear morning/evening peaks
    • **Weekday patterns** differ substantially from weekends
    • **Precipitation** has strong negative impact on bike usage
    • **Station location** matters for accessibility and convenience
    """)
    
    # SHAP-style explanation (simulated)
    st.subheader("Sample Prediction Explanation")
    st.write("**Prediction for Central Park Station at 8 AM on Tuesday:**")
    
    explanation_data = pd.DataFrame({
        'feature': ['Temperature (22°C)', 'Morning Rush Hour', 'Weekday', 'No Precipitation', 'High Traffic Area'],
        'contribution': [+15, +25, +10, +5, +8],
        'base_prediction': [50, 50, 50, 50, 50]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Positive Impact',
        x=explanation_data['feature'],
        y=explanation_data['contribution'],
        marker_color='green'
    ))
    
    fig.update_layout(
        title='Feature Contributions to Prediction',
        xaxis_title='Features',
        yaxis_title='Impact on Prediction'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 This prediction suggests high bike demand due to optimal weather and commute timing.")

# New tab for the chatbot
with tabs[4]:
    st.header("UrbanFlow Assistant")
    
   
    st.markdown("Ask me questions about bike usage, weather impact, traffic patterns, or popular stations!")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
    
    # Chat input - Use form to properly handle submission
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question here:")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get chatbot response
            response = get_chatbot_response(user_input)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the UI
            st.rerun()
    
    # Add some example questions to help users get started
    st.markdown("### Example questions:")
    example_questions = [
        "How does weather impact bike usage?",
        "What are the most popular bike stations?",
        "Is there a correlation between traffic and bike usage?",
        "When are bikes used the most during the day?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"q_{question}"):
            # Add question to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get chatbot response
            response = get_chatbot_response(question)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the UI
            st.rerun()
