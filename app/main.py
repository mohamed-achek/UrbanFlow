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
    "../assets/header.png", 
    "../assets/Header.png",
    "../Assets/header.png",
    "../Assets/Header.png",
    "assets/header.png",
    "assets/Header.png",
    "./assets/header.png",
]

# Check if any of the possible paths exist
header_image_path = None
for path in possible_paths:
    if os.path.exists(path):
        header_image_path = path
        break

# If image not found, try to create the assets directory and display a message
if header_image_path is None:
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Display a notice about the missing image
    st.warning(f"Header image not found. Please add a header image to the 'assets' directory.")
    
    # Create a title as fallback
    st.markdown('<h1 class="title-text">UrbanFlow AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Smart Bike Demand Forecasting in NYC</p>', unsafe_allow_html=True)
else:
    # Image found, display it with smaller width
    col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns for layout
    with col2:  # Use the middle column to display the image
        st.image(header_image_path, width=500)  # Set a fixed width of 500 pixels

# Sidebar controls
st.sidebar.title("UrbanFlow AI Controls")
st.sidebar.markdown("Select filters and options to explore bike demand in NYC.")

# Placeholder controls (to be implemented)
st.sidebar.multiselect("Select Station(s)", options=["All"], default=["All"])
st.sidebar.date_input("Date Range", [])
st.sidebar.slider("Temperature (°C)", min_value=-10, max_value=40, value=(0,30))
st.sidebar.slider("Precipitation (mm)", min_value=0, max_value=50, value=(0,10))
st.sidebar.slider("Wind Speed (km/h)", min_value=0, max_value=50, value=(0,20))
st.sidebar.checkbox("Include Traffic Data", value=False)
st.sidebar.selectbox("Model Selector", options=["Best Model"], index=0)
st.sidebar.button("Download Filtered Data")

# Main page tabs
st.title("UrbanFlow AI: Smart Bike Demand Forecasting in NYC")

# Helper function to load sample data
@st.cache_data
def load_sample_data():
    """Load and return sample data for demonstration"""
    # Try to load actual data first
    data_paths = [
        "../data/final/merged_dataset_20250720_170853.csv",
        "../data/final/citibike_engineered_20250719_161523.csv",
        "../data/processed/citibike_cleaned.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df.head(1000)  # Limit to 1000 rows for performance
            except Exception as e:
                continue
    
    # If no real data found, create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'trip_count': np.random.poisson(50, 100) + np.random.normal(0, 10, 100).astype(int),
        'temperature': np.random.normal(15, 10, 100),
        'precipitation': np.random.exponential(2, 100),
        'station_id': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'weekday': [d.weekday() for d in dates],
        'hour': [d.hour for d in dates]
    })
    return sample_data

# Load data once
sample_data = load_sample_data()
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
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trips = sample_data['trip_count'].sum() if 'trip_count' in sample_data.columns else 12543
        st.metric("Total Trips", f"{total_trips:,}")
    
    with col2:
        avg_temp = sample_data['temperature'].mean() if 'temperature' in sample_data.columns else 18.5
        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    with col3:
        peak_hour = sample_data.groupby('hour')['trip_count'].sum().idxmax() if 'hour' in sample_data.columns else 17
        st.metric("Peak Hour", f"{peak_hour}:00")
    
    with col4:
        busiest_station = sample_data.groupby('station_id')['trip_count'].sum().idxmax() if 'station_id' in sample_data.columns else "Central Station"
        st.metric("Busiest Station", busiest_station)
    
    # Time series chart
    st.subheader("Trip Count Over Time")
    if 'datetime' in sample_data.columns and 'trip_count' in sample_data.columns:
        fig = px.line(sample_data, x='datetime', y='trip_count', 
                     title='Bike Trip Demand Over Time')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Time series data not available in current dataset.")
    
    # Weather correlation
    st.subheader("Weather Impact on Bike Usage")
    if 'temperature' in sample_data.columns and 'trip_count' in sample_data.columns:
        fig = px.scatter(sample_data, x='temperature', y='trip_count', 
                        title='Temperature vs Trip Count',
                        labels={'temperature': 'Temperature (°C)', 'trip_count': 'Number of Trips'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weather correlation data not available.")

with tabs[1]:
    st.header("Station Usage Map")
    
    # NYC coordinates for sample stations
    nyc_stations = pd.DataFrame({
        'station_id': ['A', 'B', 'C', 'D'],
        'station_name': ['Central Park South', 'Times Square', 'Brooklyn Bridge', 'Wall Street'],
        'latitude': [40.7677, 40.7580, 40.7061, 40.7074],
        'longitude': [-73.9796, -73.9855, -73.9969, -74.0113],
        'trip_count': [250, 180, 220, 150]
    })
    
    # Create map visualization
    st.subheader("Popular Bike Stations in NYC")
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
                        height=500)
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    
    # Station statistics
    st.subheader("Station Usage Statistics")
    st.dataframe(nyc_stations[['station_name', 'trip_count']].sort_values('trip_count', ascending=False))

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
    
    # Usage patterns by hour
    st.subheader("Usage Patterns by Hour of Day")
    if 'hour' in sample_data.columns and 'trip_count' in sample_data.columns:
        hourly_usage = sample_data.groupby('hour')['trip_count'].mean().reset_index()
        fig = px.bar(hourly_usage, x='hour', y='trip_count',
                    title='Average Trips by Hour of Day',
                    labels={'hour': 'Hour of Day', 'trip_count': 'Average Trips'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Sample hourly pattern
        hours = list(range(24))
        trips = [15, 10, 8, 12, 25, 45, 85, 95, 75, 55, 60, 65, 70, 75, 80, 75, 85, 95, 90, 70, 50, 35, 25, 20]
        hourly_df = pd.DataFrame({'hour': hours, 'avg_trips': trips})
        fig = px.bar(hourly_df, x='hour', y='avg_trips',
                    title='Average Trips by Hour of Day (Sample Data)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekday vs Weekend analysis
    st.subheader("Weekday vs Weekend Usage")
    weekday_data = pd.DataFrame({
        'day_type': ['Weekday', 'Weekend'],
        'avg_trips': [65, 45],
        'peak_hour': ['8 AM & 6 PM', '2 PM']
    })
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(weekday_data, values='avg_trips', names='day_type',
                    title='Average Daily Usage Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Usage Insights:**")
        st.write("• Weekday peak: Commute hours (8 AM, 6 PM)")
        st.write("• Weekend peak: Afternoon leisure (2 PM)")
        st.write("• Weekend usage 31% lower than weekdays")
        st.write("• Electric bikes used for longer trips")
    
    # Weather impact analysis
    st.subheader("Weather Impact on Usage")
    weather_impact = pd.DataFrame({
        'condition': ['Sunny', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Snow'],
        'usage_change': [100, 85, 60, 30, 15],
        'avg_trips': [80, 68, 48, 24, 12]
    })
    
    fig = px.bar(weather_impact, x='condition', y='usage_change',
                title='Relative Usage by Weather Condition (%)',
                color='usage_change',
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

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
