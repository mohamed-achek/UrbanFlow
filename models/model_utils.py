"""
Model utilities for UrbanFlow AI
This module provides functions for model training, evaluation, and deployment
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# Deep Learning imports for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM model will not be available.")

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def prepare_features(df: pd.DataFrame, target_col: str = 'trip_count') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for machine learning
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Handle datetime columns
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        data[f'{col}_hour'] = data[col].dt.hour
        data[f'{col}_day'] = data[col].dt.day
        data[f'{col}_month'] = data[col].dt.month
        data[f'{col}_weekday'] = data[col].dt.weekday
        data[f'{col}_weekend'] = (data[col].dt.weekday >= 5).astype(int)
    
    # Drop datetime columns after feature extraction
    data = data.drop(columns=datetime_cols, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        if col != target_col:  # Don't encode target if it's categorical
            data[col] = le.fit_transform(data[col].astype(str))
    
    # Separate features and target
    if target_col in data.columns:
        X = data.drop(columns=[target_col])
        y = data[target_col]
    else:
        X = data
        y = pd.Series()
        print(f"Warning: Target column '{target_col}' not found in data")
    
    # Handle infinite and NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def get_models_dict() -> Dict[str, Any]:
    """
    Get dictionary of available models for training
    
    Returns:
        Dict[str, Any]: Dictionary of model names and instances
    """
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Add LSTM model if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        models['lstm'] = 'lstm_model'  # Special key for LSTM
    
    return models

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.inf
    
    return metrics

def create_lstm_model(input_shape: Tuple[int, int], 
                     lstm_units: int = 50, 
                     dropout_rate: float = 0.2,
                     learning_rate: float = 0.001) -> Any:
    """
    Create LSTM model for time series forecasting
    
    Args:
        input_shape (Tuple[int, int]): Shape of input data (timesteps, features)
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM model")
    
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_lstm_data(X: pd.DataFrame, y: pd.Series, 
                     sequence_length: int = 10,
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """
    Prepare data for LSTM training with time sequences
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        sequence_length (int): Length of input sequences
        test_size (float): Test set size
        
    Returns:
        Tuple of train/test data and scalers
    """
    # Scale the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_sequences, y_sequences = [], []
    
    for i in range(sequence_length, len(X_scaled)):
        X_sequences.append(X_scaled[i-sequence_length:i])
        y_sequences.append(y_scaled[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split data
    split_idx = int(len(X_sequences) * (1 - test_size))
    
    X_train = X_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_train = y_sequences[:split_idx]
    y_test = y_sequences[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def train_lstm_model(X: pd.DataFrame, y: pd.Series, 
                    sequence_length: int = 10,
                    lstm_units: int = 50,
                    epochs: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.2) -> Dict[str, Any]:
    """
    Train LSTM model for time series prediction
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        sequence_length (int): Length of input sequences
        lstm_units (int): Number of LSTM units
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Validation split ratio
        
    Returns:
        Dict containing trained model and results
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM model")
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_lstm_data(
        X, y, sequence_length
    )
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, lstm_units)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred).flatten()
    
    # Evaluate model
    metrics = evaluate_model(y_test_original, y_pred_original)
    
    return {
        'model': model,
        'metrics': metrics,
        'history': history.history,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'sequence_length': sequence_length,
        'y_test': y_test_original,
        'y_pred': y_pred_original
    }

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Dict]:
    """
    Train and evaluate multiple models
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Test set size
        
    Returns:
        Dict[str, Dict]: Results for each model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get models
    models = get_models_dict()
    results = {}
    
    print("Training and evaluating models...")
    
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            
            if name == 'lstm':
                # Special handling for LSTM model
                if TENSORFLOW_AVAILABLE:
                    lstm_results = train_lstm_model(X, y)
                    results[name] = {
                        'model': lstm_results['model'],
                        'scaler': lstm_results['feature_scaler'],
                        'target_scaler': lstm_results['target_scaler'],
                        'metrics': lstm_results['metrics'],
                        'sequence_length': lstm_results['sequence_length'],
                        'predictions': lstm_results['y_pred'],
                        'cv_rmse': lstm_results['metrics']['rmse']  # Use test RMSE as CV score
                    }
                    print(f"{name} - RMSE: {lstm_results['metrics']['rmse']:.4f}, R²: {lstm_results['metrics']['r2']:.4f}")
                else:
                    print(f"Skipping {name} - TensorFlow not available")
                continue
            
            # Train Random Forest model
            if name == 'random_forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = evaluate_model(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                results[name] = {
                    'model': model,
                    'scaler': None,
                    'metrics': metrics,
                    'cv_rmse': cv_rmse,
                    'predictions': y_pred
                }
                
                print(f"{name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    return results

def save_model(model: Any, model_name: str, scaler: Optional[Any] = None) -> str:
    """
    Save trained model to disk
    
    Args:
        model (Any): Trained model
        model_name (str): Name for the model file
        scaler (Optional[Any]): Scaler object if used
        
    Returns:
        str: Path to saved model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    try:
        joblib.dump(model_data, model_path)
        print(f"Model saved successfully: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return ""

def load_model(model_path: str) -> Tuple[Any, Optional[Any]]:
    """
    Load trained model from disk
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        Tuple[Any, Optional[Any]]: Loaded model and scaler
    """
    try:
        model_data = joblib.load(model_path)
        return model_data['model'], model_data.get('scaler')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_latest_model(model_name: str) -> Tuple[Any, Optional[Any], str]:
    """
    Get the latest saved model by name
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Tuple[Any, Optional[Any], str]: Model, scaler, and file path
    """
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_name) and f.endswith('.pkl')]
    
    if not model_files:
        print(f"No saved models found for {model_name}")
        return None, None, ""
    
    # Sort by timestamp (most recent first)
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(MODELS_DIR, model_files[0])
    
    model, scaler = load_model(latest_model_path)
    return model, scaler, latest_model_path

def load_lstm_model(model_path: str) -> Tuple[Any, Optional[Any], Optional[Any], int]:
    """
    Load LSTM model from .h5 file along with scalers and metadata
    
    Args:
        model_path (str): Path to the .h5 model file
        
    Returns:
        Tuple[Any, Optional[Any], Optional[Any], int]: Model, feature_scaler, target_scaler, sequence_length
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to load LSTM models")
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Try to load accompanying scalers and metadata
        base_path = model_path.replace('.h5', '')
        scaler_path = f"{base_path}_scalers.pkl"
        
        feature_scaler = None
        target_scaler = None
        sequence_length = 10  # Default value
        
        if os.path.exists(scaler_path):
            scaler_data = joblib.load(scaler_path)
            feature_scaler = scaler_data.get('feature_scaler')
            target_scaler = scaler_data.get('target_scaler')
            sequence_length = scaler_data.get('sequence_length', 10)
        
        print(f"Successfully loaded LSTM model from {model_path}")
        return model, feature_scaler, target_scaler, sequence_length
        
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None, None, None, 10

def save_lstm_model(model: Any, model_name: str, feature_scaler: Any, target_scaler: Any, 
                   sequence_length: int, metrics: Dict[str, float]) -> str:
    """
    Save LSTM model to .h5 file along with scalers and metadata
    
    Args:
        model: Trained LSTM model
        model_name (str): Name for the model file
        feature_scaler: Feature scaler
        target_scaler: Target scaler
        sequence_length (int): Sequence length used for training
        metrics (Dict[str, float]): Model performance metrics
        
    Returns:
        str: Path to saved model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to save LSTM models")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}_scalers.pkl")
    
    try:
        # Save the Keras model
        model.save(model_path)
        
        # Save scalers and metadata
        scaler_data = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'sequence_length': sequence_length,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        joblib.dump(scaler_data, scaler_path)
        
        print(f"LSTM model saved to {model_path}")
        print(f"Scalers saved to {scaler_path}")
        
        return model_path
        
    except Exception as e:
        print(f"Error saving LSTM model: {e}")
        return ""

def predict_demand(model: Any, X: pd.DataFrame, scaler: Optional[Any] = None) -> np.ndarray:
    """
    Make predictions using trained model
    
    Args:
        model (Any): Trained model
        X (pd.DataFrame): Features for prediction
        scaler (Optional[Any]): Scaler object if needed
        
    Returns:
        np.ndarray: Predictions
    """
    try:
        if scaler is not None:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return np.array([])

def create_time_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Create time-based features from datetime column
    
    Args:
        df (pd.DataFrame): Input dataframe
        datetime_col (str): Name of datetime column
        
    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    data = df.copy()
    
    if datetime_col in data.columns:
        # Ensure datetime format
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        # Basic time features
        data['hour'] = data[datetime_col].dt.hour
        data['day'] = data[datetime_col].dt.day
        data['month'] = data[datetime_col].dt.month
        data['year'] = data[datetime_col].dt.year
        data['weekday'] = data[datetime_col].dt.weekday
        data['weekend'] = (data[datetime_col].dt.weekday >= 5).astype(int)
        data['quarter'] = data[datetime_col].dt.quarter
        
        # Cyclical features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
        
        # Rush hour indicators
        data['morning_rush'] = ((data['hour'] >= 7) & (data['hour'] <= 9) & (data['weekday'] < 5)).astype(int)
        data['evening_rush'] = ((data['hour'] >= 17) & (data['hour'] <= 19) & (data['weekday'] < 5)).astype(int)
        
        # Season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        data['season'] = data['month'].apply(get_season)
        
    return data

def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained model
    
    Args:
        model (Any): Trained model
        feature_names (List[str]): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print("Model doesn't have feature importance or coefficients")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return pd.DataFrame()

# Example usage function
def run_model_pipeline(data_path: str, target_col: str = 'trip_count') -> Dict[str, Any]:
    """
    Run complete model training pipeline
    
    Args:
        data_path (str): Path to data file
        target_col (str): Target column name
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    print("Starting UrbanFlow AI model training pipeline...")
    
    # Load data
    df = load_data(data_path)
    if df.empty:
        return {}
    
    # Prepare features
    X, y = prepare_features(df, target_col)
    if X.empty or y.empty:
        return {}
    
    # Train models
    results = train_and_evaluate_models(X, y)
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
    best_model = results[best_model_name]['model']
    best_scaler = results[best_model_name]['scaler']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best RMSE: {results[best_model_name]['metrics']['rmse']:.4f}")
    
    # Save best model
    model_path = save_model(best_model, best_model_name, best_scaler)
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_model_path': model_path,
        'feature_names': list(X.columns)
    }

# Streamlit Integration Functions
import streamlit as st

class StreamlitModelPredictor:
    """Handle model loading and predictions for Streamlit app"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.model_metadata = None
        self.sequence_length = 10
        self.model_type = None
        self.is_loaded = False
    
    @st.cache_data
    def load_trained_model(_self, model_name: str = "random_forest") -> bool:
        """Load the trained model for Streamlit app"""
        try:
            # Check for both .pkl (Random Forest) and .h5 (LSTM) files
            pkl_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            h5_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            
            if os.path.exists(pkl_path):
                # Load Random Forest model
                with open(pkl_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle different pickle formats
                if isinstance(model_data, dict):
                    _self.model = model_data.get('model')
                    _self.scaler = model_data.get('scaler')
                    _self.feature_names = model_data.get('feature_names', [])
                    _self.model_metadata = model_data.get('metadata', {})
                else:
                    # Model was saved directly
                    _self.model = model_data
                    _self.feature_names = [
                        'temperature_2m', 'precipitation', 'wind_speed_10m', 
                        'hour', 'is_weekend', 'is_peak_hour'
                    ]
                    _self.model_metadata = {}
                
                _self.model_type = "random_forest"
                
            elif os.path.exists(h5_path) and TENSORFLOW_AVAILABLE:
                # Load LSTM model
                _self.model, _self.scaler, _self.target_scaler, _self.sequence_length = load_lstm_model(h5_path)
                
                if _self.model is None:
                    st.error(f"❌ Failed to load LSTM model from {h5_path}")
                    return False
                
                _self.feature_names = [
                    'temperature_2m', 'precipitation', 'wind_speed_10m', 
                    'hour', 'is_weekend', 'is_peak_hour'
                ]
                _self.model_metadata = {}
                _self.model_type = "lstm"
                
            else:
                st.error(f"❌ Model file not found: {pkl_path} or {h5_path}")
                return False
            
            _self.is_loaded = True
            
            # Show success message
            st.success(f"🤖 {model_name} model loaded successfully!")
            
            if _self.model_metadata:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{_self.model_metadata.get('r2_score', 'N/A'):.3f}")
                with col2:
                    st.metric("RMSE", f"{_self.model_metadata.get('rmse', 'N/A'):.2f}")
                with col3:
                    st.metric("MAE", f"{_self.model_metadata.get('mae', 'N/A'):.2f}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def make_streamlit_prediction(self, input_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Make prediction for Streamlit app"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            if self.model_type == "random_forest":
                # Random Forest prediction
                features = self._prepare_streamlit_features(input_data)
                
                # Apply scaling if scaler exists
                if self.scaler:
                    features_scaled = self.scaler.transform([features])
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    prediction = self.model.predict([features])[0]
                
                # Ensure positive predictions
                prediction = max(0, prediction)
                
                # Get additional details
                details = {
                    'prediction': float(prediction),
                    'input_features': dict(zip(self.feature_names, features)),
                    'feature_importance': self._get_feature_importance_dict(),
                    'model_confidence': self._estimate_confidence(features),
                    'model_type': 'random_forest'
                }
                
            elif self.model_type == "lstm":
                # LSTM prediction (simplified for single prediction)
                features = self._prepare_streamlit_features(input_data)
                
                # For LSTM, we need sequence data. For single prediction, we'll replicate the features
                # This is a simplified approach - in practice, you'd use historical data
                features_sequence = np.array([features] * self.sequence_length).reshape(1, self.sequence_length, -1)
                
                if self.scaler:
                    features_sequence = self.scaler.transform(features_sequence.reshape(-1, features_sequence.shape[-1])).reshape(1, self.sequence_length, -1)
                
                prediction = self.model.predict(features_sequence)[0][0]
                
                # Inverse transform if target scaler exists
                if self.target_scaler:
                    prediction = self.target_scaler.inverse_transform([[prediction]])[0][0]
                
                # Ensure positive prediction
                prediction = max(0, prediction)
                
                # Get additional details
                details = {
                    'prediction': float(prediction),
                    'input_features': dict(zip(self.feature_names, features)),
                    'model_confidence': 0.8,  # Default confidence for LSTM
                    'model_type': 'lstm',
                    'sequence_length': self.sequence_length
                }
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            return prediction, details
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _prepare_streamlit_features(self, input_data: Dict[str, Any]) -> list:
        """Prepare features for prediction"""
        # Map Streamlit inputs to model features
        feature_mapping = {
            'temperature_2m': input_data.get('temperature', 20.0),
            'precipitation': input_data.get('precipitation', 0.0),
            'wind_speed_10m': input_data.get('wind_speed', 5.0),
            'hour': input_data.get('hour', 12),
            'is_weekend': int(input_data.get('is_weekend', False)),
            'is_peak_hour': int(input_data.get('is_peak_hour', False))
        }
        
        # Create feature vector
        features = []
        for name in self.feature_names:
            if name in feature_mapping:
                features.append(feature_mapping[name])
            else:
                # Default values for missing features
                if 'temperature' in name:
                    features.append(20.0)
                elif 'precipitation' in name:
                    features.append(0.0)
                elif 'wind' in name:
                    features.append(5.0)
                elif 'hour' in name:
                    features.append(12)
                else:
                    features.append(0)
        
        return features
    
    def _get_feature_importance_dict(self) -> Dict[str, float]:
        """Get feature importance as dictionary"""
        if self.model_type == "random_forest" and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif self.model_type == "lstm":
            # LSTM doesn't have traditional feature importance, return equal weights as placeholder
            equal_weight = 1.0 / len(self.feature_names)
            return dict(zip(self.feature_names, [equal_weight] * len(self.feature_names)))
        else:
            return {}
    
    def _estimate_confidence(self, features: list) -> float:
        """Estimate prediction confidence"""
        try:
            if hasattr(self.model, 'estimators_'):
                # For ensemble models, use prediction variance
                predictions = [tree.predict([features])[0] for tree in self.model.estimators_[:10]]
                std_dev = np.std(predictions)
                mean_pred = np.mean(predictions)
                confidence = max(0.1, 1.0 - (std_dev / max(mean_pred, 1)))
                return min(0.99, confidence)
        except:
            pass
        return 0.75  # Default confidence

# Global predictor instance
streamlit_predictor = StreamlitModelPredictor()

def load_model_for_streamlit(model_name: str = "random_forest") -> bool:
    """Load model for Streamlit app"""
    return streamlit_predictor.load_trained_model(model_name)

def predict_for_streamlit(input_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Make prediction for Streamlit app"""
    return streamlit_predictor.make_streamlit_prediction(input_data)

def get_model_info_for_streamlit() -> Dict[str, Any]:
    """Get model information for Streamlit display"""
    if not streamlit_predictor.is_loaded:
        return {"status": "Not loaded"}
    
    return {
        "status": "Loaded",
        "model_type": type(streamlit_predictor.model).__name__,
        "feature_names": streamlit_predictor.feature_names,
        "metadata": streamlit_predictor.model_metadata,
        "feature_importance": streamlit_predictor._get_feature_importance_dict()
    }

if __name__ == "__main__":
    # Example usage
    print("UrbanFlow AI Model Utilities")
    print("Available functions:")
    print("- load_data()")
    print("- prepare_features()")
    print("- train_and_evaluate_models()")
    print("- save_model() / load_model()")
    print("- predict_demand()")
    print("- run_model_pipeline()")
    print("- load_model_for_streamlit()")
    print("- predict_for_streamlit()")