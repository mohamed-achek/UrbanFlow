import os
import pandas as pd
import random
from datetime import datetime
from typing import List, Optional

# Updated LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFaceHub  # Fixed import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

class UrbanFlowLangChainBot:
    def __init__(self, use_openai=False):  # Changed default to False to prefer Hugging Face
        self.name = "UrbanFlow Assistant"
        self.greeting_phrases = [
            "Hello! I'm your UrbanFlow Assistant. How can I help you?",
            "Hi there! I can answer questions about bike usage, weather, and traffic in NYC.",
            "Welcome to UrbanFlow AI! I'm here to assist with any questions about our data and predictions."
        ]
        
        # Project data paths
        self.data_path = "../data/final/"
        self.docs_path = "../docs/"
        self.use_openai = use_openai
        
        # Initialize chat components
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Setup LangChain components
        self._setup_langchain()
        
    def _setup_langchain(self):
        """Set up LangChain components based on available models"""
        try:
            # Prioritize Hugging Face models unless explicitly asked to use OpenAI
            if not self.use_openai and "HUGGINGFACEHUB_API_TOKEN" in os.environ:
                # Use Hugging Face models
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",  # Using a larger model for better responses
                    model_kwargs={"temperature": 0.5, "max_length": 512}
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"  # Better embedding model
                )
                print("Using HuggingFace models")
            # Fall back to OpenAI if requested and available
            elif self.use_openai and "OPENAI_API_KEY" in os.environ:
                self.llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
                self.embeddings = OpenAIEmbeddings()
                print("Using OpenAI models")
            # Final fallback if neither condition is met
            elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
                # Default to Hugging Face if token exists
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
                self.embeddings = HuggingFaceEmbeddings()
                print("Using HuggingFace models (fallback)")
            else:
                print("No API keys found. Unable to initialize language models.")
                return
            
            # Create the vector store and load documents
            self._load_documents()
            
        except Exception as e:
            print(f"Error setting up LangChain components: {e}")
            # If we fail, we'll fall back to the default responses
    
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
                memory=self.memory
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
                # Use LangChain for response
                response = self.conversation_chain({"question": user_input})
                return response['answer']
            except Exception as e:
                print(f"Error getting LangChain response: {e}")
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


# Initialize with environment variables - prioritize Hugging Face
has_hf_key = "HUGGINGFACEHUB_API_TOKEN" in os.environ
has_openai_key = "OPENAI_API_KEY" in os.environ

if has_hf_key:
    chatbot = UrbanFlowLangChainBot(use_openai=False)  # Explicitly use Hugging Face
elif has_openai_key:
    print("Warning: HuggingFace API token not found, falling back to OpenAI.")
    chatbot = UrbanFlowLangChainBot(use_openai=True)
else:
    # If no API keys are available, a message will be printed
    print("Warning: No API keys found for HuggingFace or OpenAI.")
    print("Please set HUGGINGFACEHUB_API_TOKEN environment variable.")
    print("Falling back to default responses without LangChain.")
    chatbot = UrbanFlowLangChainBot()

def get_chatbot_response(user_input):
    """Get a response from the chatbot for the given user input"""
    return chatbot.get_response(user_input)
