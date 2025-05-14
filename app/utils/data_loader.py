"""
Data loading and preprocessing utilities for the Supply Chain Optimizer.
Handles CSV uploads, API data fetching, and data cleaning.
"""

import pandas as pd
import numpy as np
import os
import io
import streamlit as st
import logging
from datetime import datetime
from sqlalchemy import create_engine
from app.config import DATABASE_URL, get_logger

logger = get_logger(__name__)

def load_csv_data(uploaded_file):
    """
    Load data from an uploaded CSV file
    
    Args:
        uploaded_file: A file-like object containing CSV data
        
    Returns:
        pandas.DataFrame: The processed data
    """
    try:
        # Check if the file is empty
        if uploaded_file is None:
            return None
        
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Basic data cleaning
        df = clean_dataframe(df)
        
        logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def clean_dataframe(df):
    """
    Clean and preprocess a dataframe
    
    Args:
        df: pandas.DataFrame to clean
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Handle missing values based on column type
    for col in df.columns:
        # Skip if no missing values
        if not df[col].isna().any():
            continue
            
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical/text columns
        else:
            df[col] = df[col].fillna("Unknown")
    
    # Convert date columns to datetime
    date_columns = [col for col in df.columns if any(date_str in col for date_str in ['date', 'time', 'year'])]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    return df

def detect_data_type(df):
    """
    Detect the type of supply chain data based on column names
    
    Args:
        df: pandas.DataFrame to analyze
        
    Returns:
        str: The detected data type ('procurement', 'logistics', 'supplier', or 'unknown')
    """
    column_set = set(df.columns.str.lower())
    
    # Check for procurement data
    procurement_indicators = {'product', 'item', 'purchase', 'quantity', 'cost', 'supplier'}
    if len(procurement_indicators.intersection(column_set)) >= 2:
        return 'procurement'
    
    # Check for logistics data
    logistics_indicators = {'shipment', 'transport', 'route', 'origin', 'destination', 'distance', 'weight'}
    if len(logistics_indicators.intersection(column_set)) >= 2:
        return 'logistics'
    
    # Check for supplier data
    supplier_indicators = {'supplier', 'vendor', 'location', 'rating', 'country', 'material'}
    if len(supplier_indicators.intersection(column_set)) >= 2:
        return 'supplier'
    
    # If no specific type is detected
    return 'unknown'

def load_sample_data(data_type):
    """
    Load pre-defined sample data for demonstration
    
    Args:
        data_type (str): Type of data to load ('procurement', 'logistics', or 'supplier')
        
    Returns:
        pandas.DataFrame: Sample data of the requested type
    """
    try:
        sample_path = os.path.join('data', f'sample_{data_type}.csv')
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        else:
            # If file doesn't exist, generate synthetic data
            if data_type == 'procurement':
                return generate_sample_procurement_data()
            elif data_type == 'logistics':
                return generate_sample_logistics_data()
            elif data_type == 'supplier':
                return generate_sample_supplier_data()
            else:
                raise ValueError(f"Unknown data type: {data_type}")
    
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        raise

def generate_sample_procurement_data(num_rows=100):
    """Generate synthetic procurement data for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    products = ['Electronics', 'Packaging', 'Raw Materials', 'Office Supplies', 'Chemicals']
    suppliers = ['Supplier A', 'Supplier B', 'EcoVendor', 'Global Materials Inc', 'Local Provider']
    
    df = pd.DataFrame({
        'order_id': [f'PO-{i:05d}' for i in range(1, num_rows+1)],
        'date': pd.date_range(start='2023-01-01', periods=num_rows),
        'product_category': np.random.choice(products, size=num_rows),
        'quantity': np.random.randint(10, 1000, size=num_rows),
        'unit_cost': np.round(np.random.uniform(5, 500, size=num_rows), 2),
        'supplier_name': np.random.choice(suppliers, size=num_rows),
        'country_of_origin': np.random.choice(['USA', 'China', 'Germany', 'India', 'Brazil'], size=num_rows),
        'lead_time_days': np.random.randint(7, 60, size=num_rows)
    })
    
    # Calculate total cost
    df['total_cost'] = df['quantity'] * df['unit_cost']
    
    return df

def generate_sample_logistics_data(num_rows=100):
    """Generate synthetic logistics data for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    transport_modes = ['Truck', 'Ship', 'Air', 'Rail']
    origins = ['New York', 'Shanghai', 'Hamburg', 'Mumbai', 'São Paulo']
    destinations = ['Los Angeles', 'Rotterdam', 'Sydney', 'Tokyo', 'London']
    
    df = pd.DataFrame({
        'shipment_id': [f'SH-{i:05d}' for i in range(1, num_rows+1)],
        'date': pd.date_range(start='2023-01-01', periods=num_rows),
        'transport_mode': np.random.choice(transport_modes, size=num_rows),
        'origin': np.random.choice(origins, size=num_rows),
        'destination': np.random.choice(destinations, size=num_rows),
        'distance_km': np.random.randint(500, 15000, size=num_rows),
        'weight_kg': np.random.randint(100, 10000, size=num_rows),
        'cost': np.round(np.random.uniform(500, 15000, size=num_rows), 2),
        'delivery_time_days': np.random.randint(2, 45, size=num_rows)
    })
    
    return df

def generate_sample_supplier_data(num_rows=50):
    """Generate synthetic supplier data for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    categories = ['Electronics', 'Packaging', 'Raw Materials', 'Office Supplies', 'Chemicals']
    certifications = ['ISO 14001', 'Fair Trade', 'LEED', 'None', 'B Corp']
    
    df = pd.DataFrame({
        'supplier_id': [f'SUP-{i:03d}' for i in range(1, num_rows+1)],
        'supplier_name': [f'Supplier {chr(65+i%26)}' for i in range(num_rows)],
        'category': np.random.choice(categories, size=num_rows),
        'country': np.random.choice(['USA', 'China', 'Germany', 'India', 'Brazil'], size=num_rows),
        'annual_revenue_usd': np.random.randint(100000, 10000000, size=num_rows),
        'employee_count': np.random.randint(10, 5000, size=num_rows),
        'sustainability_certification': np.random.choice(certifications, size=num_rows),
        'years_in_business': np.random.randint(1, 50, size=num_rows),
        'risk_score': np.round(np.random.uniform(1, 10, size=num_rows), 1),
        'carbon_footprint_tons': np.round(np.random.uniform(10, 5000, size=num_rows), 1),
    })
    
    return df

def save_dataframe_to_db(df, table_name):
    """
    Save a dataframe to the database
    
    Args:
        df: pandas.DataFrame to save
        table_name: Name of the table to save to
    """
    try:
        engine = create_engine(DATABASE_URL)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Successfully saved {len(df)} rows to table {table_name}")
    except Exception as e:
        logger.error(f"Error saving dataframe to database: {str(e)}")
        raise

def load_dataframe_from_db(table_name):
    """
    Load a dataframe from the database
    
    Args:
        table_name: Name of the table to load from
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql_table(table_name, engine)
        logger.info(f"Successfully loaded {len(df)} rows from table {table_name}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataframe from database: {str(e)}")
        return None
