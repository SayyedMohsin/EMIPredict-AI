import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

class MLModelTrainer:
    def __init__(self):
        self.models = {
            'classification': {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': XGBClassifier(random_state=42)
            },
            'regression': {
                'linear_regression': LinearRegression(),
                'random_forest_reg': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgboost_reg': XGBRegressor(random_state=42)
            }
        }
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, experiment_name="EMI_Classification"):
        """Train and evaluate classification models with MLflow tracking"""
        try:
            mlflow.set_experiment(experiment_name)
        except:
            print("MLflow setup failed, continuing without tracking")
        
        results = {}
        
        for name, model in self.models['classification'].items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                print(f"✅ {name} - Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
        
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, experiment_name="EMI_Regression"):
        """Train and evaluate regression models with MLflow tracking"""
        try:
            mlflow.set_experiment(experiment_name)
        except:
            print("MLflow setup failed, continuing without tracking")
        
        results = {}
        
        for name, model in self.models['regression'].items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                }
                
                print(f"✅ {name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
        
        return results