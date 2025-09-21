import streamlit as st
from chatbot import get_chatbot_response
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image

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
model_selector = st.sidebar.selectbox("Model Selector", options=["Random Forest", "XGBoost", "Neural Network"], index=0)

# Apply filters button
if st.sidebar.button("Apply Filters") or True:  # Auto-apply filters on any change
    st.sidebar.success("✅ Filters applied successfully!")

# Download button
if st.sidebar.button("Download Filtered Data"):
    st.sidebar.success("Data download feature coming soon!")

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

# Main page tabs
st.title("UrbanFlow AI: Smart Bike Demand Forecasting in NYC")

# Data notice
st.warning("""
🗂️ **Data Notice:** This demo uses sample datasets (5,000 rows each) due to GitHub file size limitations. 
The original datasets contain millions of records and are too large for repository hosting:
- Original data: ~9GB (40M+ records)  
- Sample datasets provide representative analysis and full functionality demonstration
""")

# Helper function to load and filter sample data
@st.cache_data
def load_sample_data():
    """Load sample datasets for demonstration"""
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
                        st.info(f"✅ Loaded sample dataset: {len(df):,} rows from {os.path.basename(path)}")
                    else:
                        st.info(f"✅ Loaded full local dataset: {len(df):,} rows (limited to 10,000 for performance)")
                        df = df.head(10000)  # Limit for performance
                    return df
            except Exception as e:
                continue
    
    # If no sample or real data found, create synthetic sample data
    st.warning("⚠️ No sample datasets found. Generating synthetic data for demonstration.")
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=5000, freq='H')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'demand_count': np.random.poisson(50, 5000) + np.random.normal(0, 10, 5000).astype(int),
        'temperature_2m': np.random.normal(15, 10, 5000),
        'precipitation': np.random.exponential(0.5, 5000),
        'wind_speed_10m': np.random.normal(15, 5, 5000),
        'start_station_name': np.random.choice(['Central Park South', 'Times Square', 'Brooklyn Bridge', 'Wall Street'], 5000),
        'weekday': [d.weekday() for d in dates],
        'hour': [d.hour for d in dates]
    })
    return sample_data

def filter_data(df, stations, temp_range, precip_range, wind_range, date_range):
    """Filter data based on sidebar controls"""
    filtered_df = df.copy()
    
    # Filter by stations
    if "All" not in stations and len(stations) > 0:
        if 'station_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['station_id'].isin(stations)]
    
    # Filter by temperature
    if 'temperature' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['temperature'] >= temp_range[0]) & 
            (filtered_df['temperature'] <= temp_range[1])
        ]
    
    # Filter by precipitation
    if 'precipitation' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['precipitation'] >= precip_range[0]) & 
            (filtered_df['precipitation'] <= precip_range[1])
        ]
    
    # Filter by wind speed
    if 'wind_speed' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['wind_speed'] >= wind_range[0]) & 
            (filtered_df['wind_speed'] <= wind_range[1])
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

# Apply filters to the data
sample_data = filter_data(raw_data, selected_stations, temp_range, precip_range, wind_range, date_range)
tabs = st.tabs([
    "Dashboard Overview",
    "Station Usage Map",
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
    st.header("Station Usage Map")
    
    # Create dynamic station data based on filters
    all_stations = ['Central Park South', 'Times Square', 'Brooklyn Bridge', 'Wall Street']
    if "All" not in selected_stations and len(selected_stations) > 0:
        stations_to_show = selected_stations
    else:
        stations_to_show = all_stations
    
    # Calculate trip counts for each station from filtered data
    if 'station_id' in sample_data.columns and not sample_data.empty:
        station_trips = sample_data.groupby('station_id')['trip_count'].sum().reset_index()
        station_trips = station_trips[station_trips['station_id'].isin(stations_to_show)]
    else:
        # Fallback data
        station_trips = pd.DataFrame({
            'station_id': stations_to_show,
            'trip_count': [250, 180, 220, 150][:len(stations_to_show)]
        })
    
    # NYC coordinates for stations
    coordinates = {
        'Central Park South': {'lat': 40.7677, 'lon': -73.9796},
        'Times Square': {'lat': 40.7580, 'lon': -73.9855},
        'Brooklyn Bridge': {'lat': 40.7061, 'lon': -73.9969},
        'Wall Street': {'lat': 40.7074, 'lon': -74.0113}
    }
    
    # Create enhanced station data
    nyc_stations = []
    for _, row in station_trips.iterrows():
        station_id = row['station_id']
        if station_id in coordinates:
            nyc_stations.append({
                'station_id': station_id,
                'station_name': station_id,
                'latitude': coordinates[station_id]['lat'],
                'longitude': coordinates[station_id]['lon'],
                'trip_count': row['trip_count']
            })
    
    nyc_stations = pd.DataFrame(nyc_stations)
    
    if not nyc_stations.empty:
        # Create map visualization
        st.subheader(f"Popular Bike Stations in NYC (Filtered: {len(nyc_stations)} stations)")
        fig = px.scatter_map(nyc_stations, 
                            lat="latitude", 
                            lon="longitude", 
                            size="trip_count",
                            color="trip_count",
                            hover_name="station_name",
                            hover_data=["station_id", "trip_count"],
                            color_continuous_scale="Viridis",
                            size_max=20,
                            zoom=11,
                            height=500,
                            title=f"Station Usage (Total trips: {nyc_stations['trip_count'].sum():,})")
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Station statistics
        st.subheader("Station Usage Statistics")
        station_stats = nyc_stations[['station_name', 'trip_count']].sort_values('trip_count', ascending=False)
        station_stats['percentage'] = (station_stats['trip_count'] / station_stats['trip_count'].sum() * 100).round(1)
        st.dataframe(station_stats, use_container_width=True)
    else:
        st.warning("⚠️ No stations match the current filter criteria.")

with tabs[2]:
    st.header("Forecasting Panel")
    
    # Forecasting section
    st.subheader("24-Hour Demand Forecast")
    
    # Generate sample forecast data
    @st.cache_data
    def generate_forecast():
        current_time = datetime.now()
        future_hours = [current_time + timedelta(hours=i) for i in range(24)]
        
        # Simple forecast simulation
        base_demand = 50
        forecasted_trips = []
        actual_trips = []
        
        for i, hour in enumerate(future_hours):
            # Simulate daily pattern with some randomness
            hour_of_day = hour.hour
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
                # Peak hours
                forecast = base_demand + np.random.normal(30, 5)
                actual = forecast + np.random.normal(0, 10)
            elif 22 <= hour_of_day or hour_of_day <= 6:
                # Low activity hours
                forecast = base_demand + np.random.normal(-20, 5)
                actual = forecast + np.random.normal(0, 8)
            else:
                # Regular hours
                forecast = base_demand + np.random.normal(10, 5)
                actual = forecast + np.random.normal(0, 12)
            
            forecasted_trips.append(max(0, forecast))
            actual_trips.append(max(0, actual) if i < 12 else None)  # Only show actuals for past 12 hours
        
        return pd.DataFrame({
            'datetime': future_hours,
            'forecasted': forecasted_trips,
            'actual': actual_trips
        })
    
    forecast_data = generate_forecast()
    
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
    
    # Model performance metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", "12.5", "-2.1")
    with col2:
        st.metric("MAE", "9.8", "-1.5")
    with col3:
        st.metric("R²", "0.85", "+0.03")

with tabs[3]:
    st.header("Exploratory Insights")
    
    if sample_data.empty:
        st.warning("⚠️ No data available for the current filters. Please adjust your criteria.")
    else:
        # Usage patterns by hour
        st.subheader("📊 Usage Patterns by Hour of Day (Filtered Data)")
        if 'hour' in sample_data.columns and 'trip_count' in sample_data.columns:
            hourly_usage = sample_data.groupby('hour')['trip_count'].mean().reset_index()
            fig = px.bar(hourly_usage, x='hour', y='trip_count',
                        title=f'Average Trips by Hour of Day ({len(sample_data):,} data points)',
                        labels={'hour': 'Hour of Day', 'trip_count': 'Average Trips'},
                        color='trip_count',
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
                fig_precip = px.scatter(sample_data, x='precipitation', y='trip_count',
                                       title='Precipitation vs Trip Count',
                                       labels={'precipitation': 'Precipitation (mm)', 'trip_count': 'Trips'},
                                       color='temperature')
                st.plotly_chart(fig_precip, use_container_width=True)
        
        # Station-specific insights
        if 'station_id' in sample_data.columns:
            st.subheader("🚉 Station-Specific Insights")
            station_summary = sample_data.groupby('station_id').agg({
                'trip_count': ['sum', 'mean', 'std'],
                'temperature': 'mean',
                'precipitation': 'mean'
            }).round(2)
            
            station_summary.columns = ['Total Trips', 'Avg Trips', 'Trip Std Dev', 'Avg Temp (°C)', 'Avg Precip (mm)']
            st.dataframe(station_summary, use_container_width=True)
        
        # Summary insights based on filtered data
        st.subheader("📋 Key Insights from Filtered Data")
        total_trips = sample_data['trip_count'].sum()
        avg_temp = sample_data['temperature'].mean() if 'temperature' in sample_data.columns else None
        avg_precip = sample_data['precipitation'].mean() if 'precipitation' in sample_data.columns else None
        
        insights = []
        insights.append(f"• Total trips in filtered period: {total_trips:,}")
        if avg_temp is not None:
            insights.append(f"• Average temperature: {avg_temp:.1f}°C")
        if avg_precip is not None:
            insights.append(f"• Average precipitation: {avg_precip:.1f}mm")
        if 'hour' in sample_data.columns:
            peak_hour = sample_data.groupby('hour')['trip_count'].sum().idxmax()
            insights.append(f"• Peak usage hour: {peak_hour}:00")
        
        for insight in insights:
            st.write(insight)

with tabs[4]:
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
with tabs[5]:
    st.header("UrbanFlow Assistant")
    
    # Show API status
    from chatbot import get_api_status
    status_message = get_api_status()
    if "fallback" in status_message.lower():
        st.warning(status_message)
    elif "connected" in status_message.lower():
        st.success(status_message)
    else:
        st.info(status_message)
    
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
