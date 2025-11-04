import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_advanced_features(self, df):
        """Create comprehensive feature set"""
        features = df.copy()
        
        # Financial Stability Features
        if all(col in df.columns for col in ['bank_balance', 'monthly_salary']):
            features['savings_ratio'] = features['bank_balance'] / features['monthly_salary']
        
        if all(col in df.columns for col in ['emergency_fund', 'monthly_salary']):
            features['emergency_coverage'] = features['emergency_fund'] / features['monthly_salary']
        
        # Loan Burden Features
        if all(col in df.columns for col in ['current_emi_amount', 'monthly_salary']):
            features['existing_loan_burden'] = features['current_emi_amount'] / features['monthly_salary']
        
        return features