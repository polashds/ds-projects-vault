# src/model.py
import pickle
import numpy as np
from typing import List, Dict, Any

class MLModel:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make prediction on input features"""
        try:
            # Convert to numpy array and reshape for single sample
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0]
            
            return {
                'prediction': int(prediction),
                'probabilities': probability.tolist(),
                'confidence': float(max(probability))
            }
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def batch_predict(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        """Make predictions on multiple samples"""
        return [self.predict(features) for features in features_list]