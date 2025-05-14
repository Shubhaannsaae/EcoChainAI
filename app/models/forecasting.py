"""
Forecasting models for predicting future emissions and supply chain metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

from app.config import get_logger

logger = get_logger(__name__)

def preprocess_time_series_data(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    freq: str = 'M'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess time series data for forecasting
    
    Args:
        df: DataFrame with time series data
        date_column: Name of the date column
        target_column: Name of the target variable column
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Tuple containing:
        - Preprocessed DataFrame
        - List of feature column names
    """
    # Copy dataframe to avoid modifying original
    df_copy = df.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df_copy[date_column]):
        try:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        except Exception as e:
            logger.error(f"Error converting date column to datetime: {str(e)}")
            raise ValueError(f"Could not convert {date_column} to datetime")
    
    # Set date as index
    df_copy.set_index(date_column, inplace=True)
    
    # Resample to specified frequency
    df_resampled = df_copy[target_column].resample(freq).sum()
    
    # Reset index to get the date as a column again
    df_resampled = df_resampled.reset_index()
    
    # Create time-based features
    df_resampled['year'] = df_resampled[date_column].dt.year
    df_resampled['month'] = df_resampled[date_column].dt.month
    df_resampled['quarter'] = df_resampled[date_column].dt.quarter
    
    # Add trend feature
    df_resampled['trend'] = np.arange(len(df_resampled))
    
    # Add lagged features
    for lag in [1, 2, 3]:
        df_resampled[f'lag_{lag}'] = df_resampled[target_column].shift(lag)
    
    # Add rolling average
    df_resampled['rolling_avg_3'] = df_resampled[target_column].rolling(window=3).mean()
    
    # Drop missing values (will be at the start due to lagging)
    df_resampled.dropna(inplace=True)
    
    # Feature columns
    feature_columns = ['year', 'month', 'quarter', 'trend', 'lag_1', 'lag_2', 'lag_3', 'rolling_avg_3']
    
    return df_resampled, feature_columns

def train_forecast_model(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    freq: str = 'M',
    model_type: str = 'random_forest'
) -> Tuple[Any, StandardScaler, StandardScaler, Dict[str, float]]:
    """
    Train a forecasting model
    
    Args:
        df: DataFrame with time series data
        date_column: Name of the date column
        target_column: Name of the target variable column
        freq: Frequency for resampling
        model_type: Type of model to train ('linear' or 'random_forest')
        
    Returns:
        Tuple containing:
        - Trained model
        - Feature scaler
        - Target scaler
        - Dictionary with model performance metrics
    """
    try:
        # Preprocess data
        df_processed, feature_columns = preprocess_time_series_data(
            df=df,
            date_column=date_column,
            target_column=target_column,
            freq=freq
        )
        
        if len(df_processed) < 10:
            logger.warning(f"Not enough data for forecasting. Need at least 10 data points, got {len(df_processed)}")
            raise ValueError("Not enough data for forecasting")
        
        # Split data into features and target
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features and target
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Select and train model
        if model_type == 'linear':
            model = LinearRegression()
        else:  # default to random forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate model
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Trained {model_type} forecasting model. MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return model, feature_scaler, target_scaler, metrics
        
    except Exception as e:
        logger.error(f"Error training forecast model: {str(e)}")
        raise

def generate_forecast(
    model: Any,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    last_data: pd.DataFrame,
    periods: int = 12,
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Generate forecast for future periods
    
    Args:
        model: Trained forecasting model
        feature_scaler: StandardScaler for features
        target_scaler: StandardScaler for target
        last_data: DataFrame with most recent data
        periods: Number of periods to forecast
        freq: Frequency of forecast
        
    Returns:
        DataFrame with forecast results
    """
    try:
        # Get the last date in the data
        last_date = last_data.index.max()
        
        # Create a list to store forecasts
        forecasts = []
        
        # Get the last values for lagged features
        last_target_values = last_data.iloc[-3:].values.flatten()
        last_target_values = last_target_values[::-1]  # Reverse to get most recent first
        
        # Pad if we don't have enough history
        if len(last_target_values) < 3:
            padding = [last_target_values[0]] * (3 - len(last_target_values))
            last_target_values = np.append(last_target_values, padding)
        
        # Last rolling average
        rolling_avg = np.mean(last_target_values[:3])
        
        # Generate forecasts for each future period
        for i in range(periods):
            # Calculate the date for this forecast period
            if freq == 'M':
                forecast_date = last_date + pd.DateOffset(months=i+1)
            elif freq == 'W':
                forecast_date = last_date + pd.DateOffset(weeks=i+1)
            else:  # Default to daily
                forecast_date = last_date + pd.DateOffset(days=i+1)
            
            # Create feature values for this period
            features = {
                'year': forecast_date.year,
                'month': forecast_date.month,
                'quarter': (forecast_date.month - 1) // 3 + 1,
                'trend': len(last_data) + i,
                'lag_1': last_target_values[0],
                'lag_2': last_target_values[1],
                'lag_3': last_target_values[2],
                'rolling_avg_3': rolling_avg
            }
            
            # Create a dataframe with the features
            X_forecast = pd.DataFrame([features])
            
            # Scale features
            X_forecast_scaled = feature_scaler.transform(X_forecast)
            
            # Make prediction
            y_forecast_scaled = model.predict(X_forecast_scaled)
            y_forecast = target_scaler.inverse_transform(y_forecast_scaled.reshape(-1, 1)).flatten()[0]
            
            # Update lagged values for next iteration
            last_target_values = np.append([y_forecast], last_target_values[:2])
            rolling_avg = np.mean(last_target_values[:3])
            
            # Save forecast
            forecasts.append({
                'date': forecast_date,
                'forecast': y_forecast
            })
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts)
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise

def forecast_emissions(
    emissions_data: pd.DataFrame,
    date_column: str,
    emissions_column: str,
    periods: int = 12,
    freq: str = 'M'
) -> Dict[str, Any]:
    """
    Forecast future emissions based on historical data
    
    Args:
        emissions_data: DataFrame with historical emissions data
        date_column: Name of the date column
        emissions_column: Name of the emissions column
        periods: Number of periods to forecast
        freq: Frequency of forecast ('M' for monthly, 'W' for weekly, 'D' for daily)
        
    Returns:
        Dictionary with forecast results and metrics
    """
    try:
        # Check if we have enough data
        if len(emissions_data) < 10:
            logger.warning(f"Not enough data for forecasting. Using trend extrapolation instead.")
            return _simple_trend_extrapolation(emissions_data, date_column, emissions_column, periods, freq)
        
        # Train model
        model, feature_scaler, target_scaler, metrics = train_forecast_model(
            df=emissions_data,
            date_column=date_column,
            target_column=emissions_column,
            freq=freq,
            model_type='random_forest'
        )
        
        # Preprocess data for forecasting
        data_processed, _ = preprocess_time_series_data(
            df=emissions_data,
            date_column=date_column,
            target_column=emissions_column,
            freq=freq
        )
        
        # Generate forecast
        forecast_df = generate_forecast(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            last_data=data_processed,
            periods=periods,
            freq=freq
        )
        
        # Calculate BAU (business as usual) emissions
        bau_total = forecast_df['forecast'].sum()
        
        # Calculate optimized emissions (assuming 15% reduction)
        optimized_forecast = forecast_df.copy()
        optimized_forecast['optimized_forecast'] = optimized_forecast['forecast'] * 0.85
        optimized_total = optimized_forecast['optimized_forecast'].sum()
        
        # Create result dictionary
        result = {
            'forecast_data': forecast_df.to_dict('records'),
            'bau_total': bau_total,
            'optimized_total': optimized_total,
            'reduction_potential': bau_total - optimized_total,
            'reduction_percent': 15.0,
            'metrics': metrics
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error forecasting emissions: {str(e)}")
        
        # Fallback to simple trend extrapolation
        logger.info("Falling back to simple trend extrapolation")
        return _simple_trend_extrapolation(emissions_data, date_column, emissions_column, periods, freq)

def _simple_trend_extrapolation(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int,
    freq: str
) -> Dict[str, Any]:
    """
    Simple trend extrapolation when not enough data is available for ML forecasting
    
    Args:
        data: DataFrame with historical data
        date_column: Name of the date column
        value_column: Name of the value column
        periods: Number of periods to forecast
        freq: Frequency of forecast
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Copy dataframe
        df = data.copy()
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Calculate average value
        avg_value = df[value_column].mean()
        
        # Calculate simple trend (average month-to-month change)
        if len(df) > 1:
            df = df.sort_values(date_column)
            values = df[value_column].values
            trend = (values[-1] - values[0]) / (len(values) - 1)
        else:
            trend = 0  # No trend if only one data point
        
        # Get last date
        last_date = df[date_column].max()
        
        # Generate forecast dates
        forecast_dates = []
        for i in range(periods):
            if freq == 'M':
                forecast_dates.append(last_date + pd.DateOffset(months=i+1))
            elif freq == 'W':
                forecast_dates.append(last_date + pd.DateOffset(weeks=i+1))
            else:  # Default to daily
                forecast_dates.append(last_date + pd.DateOffset(days=i+1))
        
        # Generate forecasts
        forecasts = []
        last_value = df[value_column].iloc[-1] if not df.empty else avg_value
        
        for i, date in enumerate(forecast_dates):
            forecast_value = last_value + trend * (i + 1)
            # Ensure forecast is not negative
            forecast_value = max(forecast_value, 0)
            
            forecasts.append({
                'date': date,
                'forecast': forecast_value
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Calculate BAU (business as usual) total
        bau_total = forecast_df['forecast'].sum()
        
        # Calculate optimized total (assuming 15% reduction)
        optimized_total = bau_total * 0.85
        
        # Create result dictionary
        result = {
            'forecast_data': forecast_df.to_dict('records'),
            'bau_total': bau_total,
            'optimized_total': optimized_total,
            'reduction_potential': bau_total - optimized_total,
            'reduction_percent': 15.0,
            'metrics': {
                'note': 'Simple trend extrapolation used due to limited data'
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error performing simple trend extrapolation: {str(e)}")
        
        # Return an empty forecast as last resort
        empty_result = {
            'forecast_data': [],
            'bau_total': 0,
            'optimized_total': 0,
            'reduction_potential': 0,
            'reduction_percent': 0,
            'metrics': {
                'error': str(e)
            }
        }
        
        return empty_result
