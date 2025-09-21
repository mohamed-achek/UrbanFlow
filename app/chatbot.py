import os
import pandas as pd
import random
from datetime import datetime
from typing import List, Optional

# Add dotenv to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from different possible locations
    load_dotenv()  # Load from .env in current directory
    load_dotenv("../.env")  # Try one directory up
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Cannot load .env file.")
    print("Install with: pip install python-dotenv")
except Exception as e:
    print(f"Error loading .env file: {e}")

# Import LangChain components with proper error handling
try:
    # First try importing from the newer langchain_huggingface package
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using HuggingFaceEmbeddings from langchain_huggingface")
    
    # For the LLM, we'll try a different approach to avoid provider errors
    from langchain_community.llms import HuggingFaceHub  # Use HuggingFaceHub instead of HuggingFaceEndpoint
    print("Using HuggingFaceHub from langchain_community")
except ImportError:
    # Fall back to community imports if needed
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub
    print("Using HuggingFace components from langchain_community")

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

class UrbanFlowLangChainBot:
    def __init__(self):  # Removed use_openai parameter
        self.name = "UrbanFlow Assistant"
        self.greeting_phrases = [
            "Hello! I'm your UrbanFlow Assistant. How can I help you?",
            "Hi there! I can answer questions about bike usage, weather, and traffic in NYC.",
            "Welcome to UrbanFlow AI! I'm here to assist with any questions about our data and predictions."
        ]
        
        # Project data paths
        self.data_path = "../data/final/"
        self.docs_path = "../docs/"
        
        # Initialize chat components
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.conversation_chain = None
        
        # Add status tracking for API key and fallback
        self.is_using_fallback = False
        self.api_key_message = ""
        
        # Update memory implementation to fix the deprecation warning
        from langchain.schema import messages_from_dict, messages_to_dict
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Setup LangChain components
        self._setup_langchain()
        
    def _setup_langchain(self):
        """Set up LangChain components based on available models"""
        # Check for API key first
        if not self._check_api_key():
            # If no API key, we'll use fallback responses
            self.is_using_fallback = True
            return
            
        try:
            # Switch to HuggingFaceHub which is more stable than HuggingFaceEndpoint
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",  # Using a smaller model that's more likely to work
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                model_kwargs={
                    "temperature": 0.5,
                    "max_length": 512
                },
                task="text2text-generation"  # Specify the task to fix validation error
            )
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Using a smaller, faster embedding model
            )
            print("Using HuggingFace models")
            
            # Create the vector store and load documents
            self._load_documents()
            
        except Exception as e:
            print(f"Error setting up LangChain components: {e}")
            self.is_using_fallback = True
            self.api_key_message = f"Error initializing models: {str(e)}"
    
    def _check_api_key(self):
        """Check if the HuggingFace API key is available"""
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            self.api_key_message = (
                "No HuggingFace API key found. Please set the HUGGINGFACEHUB_API_TOKEN "
                "environment variable. Using basic responses without AI capabilities."
            )
            print(self.api_key_message)
            return False
        elif not os.environ["HUGGINGFACEHUB_API_TOKEN"].strip():
            self.api_key_message = (
                "HuggingFace API key is empty. Please set a valid API key in the "
                "HUGGINGFACEHUB_API_TOKEN environment variable."
            )
            print(self.api_key_message)
            return False
        return True
    
    def _load_documents(self):
        """Load project documents and create vector store"""
        documents = []
        
        # 1. Load CSV data files if they exist
        try:
            if os.path.exists(self.data_path):
                # Load bike data
                bike_path = os.path.join(self.data_path, "citibike_engineered.csv")
                if os.path.exists(bike_path):
                    loader = CSVLoader(file_path=bike_path)
                    documents.extend(loader.load())
                    print(f"Loaded bike data from {bike_path}")
                
                # Load weather data
                weather_path = os.path.join(self.data_path, "weather_engineered.csv")
                if os.path.exists(weather_path):
                    loader = CSVLoader(file_path=weather_path)
                    documents.extend(loader.load())
                    print(f"Loaded weather data from {weather_path}")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            
        # 2. Load project documentation if it exists
        try:
            if os.path.exists(self.docs_path):
                # Load text files
                txt_loader = DirectoryLoader(self.docs_path, glob="**/*.txt", loader_cls=TextLoader)
                documents.extend(txt_loader.load())
                
                # Load PDF files
                pdf_loader = DirectoryLoader(self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents.extend(pdf_loader.load())
                
                print(f"Loaded documentation from {self.docs_path}")
        except Exception as e:
            print(f"Error loading documentation: {e}")
        
        # 3. Add project-specific information as documents
        project_info = [
            Document(
                page_content="UrbanFlow AI is a smart bike demand forecasting system for NYC. "
                            "It analyzes Citi Bike usage patterns in relation to weather, traffic, and temporal factors. "
                            "The system uses machine learning to predict bike demand across different stations and conditions.",
                metadata={"source": "project_description"}
            ),
            Document(
                page_content="Weather impacts bike usage significantly. Temperature is the strongest predictor, with "
                            "precipitation decreasing bike usage by approximately 40%. Ideal biking conditions are "
                            "between 15-25°C with no precipitation.",
                metadata={"source": "weather_impact"}
            ),
            Document(
                page_content="Bike usage patterns differ between weekdays and weekends. On weekdays, usage peaks during "
                            "commute hours (7-9am and 5-7pm). Weekend usage is more evenly distributed throughout the day "
                            "with a higher concentration in recreational areas.",
                metadata={"source": "usage_patterns"}
            ),
            Document(
                page_content="Electric bikes tend to be used for longer trips compared to classic bikes. The average "
                            "trip distance for electric bikes is 2.5km, while for classic bikes it's 1.8km.",
                metadata={"source": "bike_types"}
            ),
        ]
        
        documents.extend(project_info)
        
        if documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # Create conversation chain
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=False,
                verbose=False
            )
            
            print(f"Created vector store with {len(texts)} text chunks")
        else:
            print("No documents found to create vector store")
    
    def get_greeting(self):
        """Return a random greeting message"""
        return random.choice(self.greeting_phrases)
    
    def get_response(self, user_input: str) -> str:
        """Get a response using the LangChain conversation chain"""
        if not user_input:
            return self.get_greeting()
            
        if self.conversation_chain:
            try:
                # Use LangChain for response - update to use invoke() method
                response = self.conversation_chain.invoke({"question": user_input})
                return response['answer']
            except Exception as e:
                # Improved error handling with more details
                import traceback
                error_details = traceback.format_exc()
                print(f"Error getting LangChain response: {e}")
                print(f"Error details: {error_details}")
                
                # Fall back to default response
                return self._get_default_response(user_input)
        else:
            # If LangChain setup failed, use default responses
            return self._get_default_response(user_input)
    
    def _get_default_response(self, user_input: str) -> str:
        """Fallback responses when LangChain is not available"""
        user_input = user_input.lower()
        
        # Default knowledge base with predefined responses
        knowledge_base = {
            "bike usage": [
                "Our data shows bikes are most popular during morning and evening commute hours.",
                "Electric bikes tend to be used for longer trips compared to classic bikes.",
                "Weekend bike usage patterns differ significantly from weekday patterns."
            ],
            "weather impact": [
                "Temperature is the strongest weather predictor of bike usage.",
                "Precipitation decreases bike usage by approximately 40%.",
                "Ideal biking conditions are between 15-25°C with no precipitation."
            ],
            "traffic": [
                "Higher traffic volumes correlate with increased bike usage in certain areas.",
                "Morning rush hour sees the highest competition between bikes and vehicles.",
                "Bike lanes in high-traffic areas show consistently higher utilization."
            ],
            "popular stations": [
                "The most popular stations are typically near major transit hubs.",
                "Stations near parks see higher weekend usage.",
                "Midtown Manhattan has the highest density of popular stations."
            ],
            "help": [
                "I can answer questions about bike usage patterns, weather impacts, traffic correlations, and station popularity. What would you like to know?"
            ]
        }
        
        # Check if input matches any keywords in knowledge base
        for key, responses in knowledge_base.items():
            if key in user_input:
                return random.choice(responses)
        
        # If input contains a greeting, respond with a greeting
        if any(greeting in user_input for greeting in ["hello", "hi", "hey", "greetings"]):
            return self.get_greeting()
        
        # Default response
        return "I'm not sure about that. You can ask me about bike usage, weather impact, traffic patterns, or popular stations."

    def get_api_status(self):
        """Get the current API connection status"""
        if self.is_using_fallback:
            if self.api_key_message:
                return f"Using fallback mode: {self.api_key_message}"
            else:
                return "Using fallback mode: No API key configured"
        else:
            return "Successfully connected to HuggingFace API"


# Initialize with environment variables - using only HuggingFace
has_hf_key = "HUGGINGFACEHUB_API_TOKEN" in os.environ

chatbot = UrbanFlowLangChainBot()

def get_chatbot_response(user_input):
    """Get a response from the chatbot for the given user input"""
    return chatbot.get_response(user_input)

def get_api_status():
    """Get the API connection status for display in the UI"""
    return chatbot.get_api_status()
