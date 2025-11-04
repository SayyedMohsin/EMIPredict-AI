import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('data/emi_prediction_dataset.xlsx')
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def data_quality_report(df):
    """Generate comprehensive data quality report"""
    report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return report