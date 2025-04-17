import pandas as pd
import os
import streamlit as st

# Define data directory
DATA_DIR = "data"

def setup_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return DATA_DIR

def import_excel_to_dataframe(file):
    """Import Excel file to pandas DataFrame"""
    try:
        df = pd.read_excel(file)
        # Check if required columns exist
        required_columns = ['phrase', 'stage', 'freq', 'success']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Calculate success_rate if not present
        if 'success_rate' not in df.columns:
            df['success_rate'] = df['success'] / df['freq']
            df['success_rate'] = df['success_rate'].fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Error importing Excel file: {str(e)}")
        return None

def save_dataframe_to_permanent_storage(df, filename="sales_pitches.csv"):
    """Save DataFrame to permanent storage"""
    setup_data_directory()
    file_path = os.path.join(DATA_DIR, filename)
    df.to_csv(file_path, index=False)
    return file_path

def load_dataframe_from_permanent_storage(filename="sales_pitches.csv"):
    """Load DataFrame from permanent storage"""
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def list_saved_datasets():
    """List all saved datasets in the data directory"""
    setup_data_directory()
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    return files 