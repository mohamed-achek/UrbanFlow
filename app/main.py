import streamlit as st
from chatbot import get_chatbot_response
import os
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
    st.info("KPIs, time series, and weather overlays will appear here.")

with tabs[1]:
    st.header("Station Usage Map")
    st.info("Interactive map of stations with trip volume.")

with tabs[2]:
    st.header("Forecasting Panel")
    st.info("ML model predictions vs actuals will be shown here.")

with tabs[3]:
    st.header("Exploratory Insights")
    st.info("EDA plots, trip durations, and demand trends.")

with tabs[4]:
    st.header("AI & Explainability")
    st.info("Vehicle count from images and SHAP/LIME explainability.")

# New tab for the chatbot
with tabs[5]:
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
    
    # Chat input
    user_input = st.text_input("Type your question here:", key="user_question")
    
    # Process user input when Enter is pressed
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        response = get_chatbot_response(user_input)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input box after processing
        st.experimental_rerun()
    
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
            st.experimental_rerun()
