import streamlit as st
from chatbot import get_chatbot_response

st.set_page_config(
    page_title="UrbanFlow AI: Smart Bike Demand Forecasting in NYC",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
