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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

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
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
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
            
            # Train model
            if name in ['linear_regression', 'ridge', 'lasso']:
                # Linear models work better with scaled features
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'scaler': scaler if name in ['linear_regression', 'ridge', 'lasso'] else None,
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