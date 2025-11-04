import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df):
        """Complete data preprocessing pipeline"""
        data = df.copy()
        
        # Convert data types first
        data = self._convert_data_types(data)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering
        data = self._feature_engineering(data)
        
        return data
    
    def _convert_data_types(self, data):
        """Convert columns to proper data types"""
        # Identify numerical columns and convert them
        numerical_columns = [
            'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
            'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
            'requested_amount', 'requested_tenure', 'max_monthly_emi'
        ]
        
        for col in numerical_columns:
            if col in data.columns:
                # Convert to numeric, coerce errors to NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values appropriately"""
        # Numerical columns - fill with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
                else:
                    data[col].fillna('Unknown', inplace=True)
        
        return data
    
    def _feature_engineering(self, data):
        """Create advanced financial features"""
        try:
            # Financial ratios - with safety checks
            if all(col in data.columns for col in ['current_emi_amount', 'monthly_salary']):
                # Ensure both are numeric
                data['current_emi_amount'] = pd.to_numeric(data['current_emi_amount'], errors='coerce')
                data['monthly_salary'] = pd.to_numeric(data['monthly_salary'], errors='coerce')
                
                # Calculate ratio with zero division handling
                data['debt_to_income_ratio'] = np.where(
                    data['monthly_salary'] > 0,
                    data['current_emi_amount'] / data['monthly_salary'],
                    0  # Default value if monthly_salary is 0
                )
            
            # Expense to income ratio
            if all(col in data.columns for col in ['monthly_rent', 'travel_expenses', 'groceries_utilities', 'monthly_salary']):
                data['monthly_rent'] = pd.to_numeric(data['monthly_rent'], errors='coerce')
                data['travel_expenses'] = pd.to_numeric(data['travel_expenses'], errors='coerce')
                data['groceries_utilities'] = pd.to_numeric(data['groceries_utilities'], errors='coerce')
                data['monthly_salary'] = pd.to_numeric(data['monthly_salary'], errors='coerce')
                
                total_expenses = data['monthly_rent'] + data['travel_expenses'] + data['groceries_utilities']
                data['expense_to_income_ratio'] = np.where(
                    data['monthly_salary'] > 0,
                    total_expenses / data['monthly_salary'],
                    0
                )
            
            # Affordability score
            if all(col in data.columns for col in ['monthly_salary', 'monthly_rent', 'travel_expenses', 'groceries_utilities', 'current_emi_amount']):
                disposable_income = (
                    data['monthly_salary'] - 
                    data['monthly_rent'] - 
                    data['travel_expenses'] - 
                    data['groceries_utilities'] - 
                    data['current_emi_amount']
                )
                data['affordability_score'] = np.where(
                    data['monthly_salary'] > 0,
                    disposable_income / data['monthly_salary'],
                    0
                )
            
            # Risk indicators
            if 'credit_score' in data.columns and 'debt_to_income_ratio' in data.columns:
                data['credit_score'] = pd.to_numeric(data['credit_score'], errors='coerce')
                data['high_risk_indicator'] = (
                    (data['credit_score'] < 600) | 
                    (data['debt_to_income_ratio'] > 0.5)
                ).astype(int)
            
            return data
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return data