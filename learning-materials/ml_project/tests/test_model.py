# tests/test_model.py
import pytest
import numpy as np
from src.model import MLModel
import pickle
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_model(tmp_path):
    """Create a sample model for testing"""
    # Create and save a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(100, 3)
    y = np.random.choice([0, 1], 100)
    model.fit(X, y)
    
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return str(model_path)

def test_model_loading(sample_model):
    """Test that model loads correctly"""
    ml_model = MLModel(sample_model)
    assert ml_model.model is not None

def test_prediction(sample_model):
    """Test single prediction"""
    ml_model = MLModel(sample_model)
    features = [0.5, -0.5, 1.0]
    
    result = ml_model.predict(features)
    
    assert 'prediction' in result
    assert 'probabilities' in result
    assert 'confidence' in result
    assert isinstance(result['prediction'], int)
    assert len(result['probabilities']) == 2

def test_batch_prediction(sample_model):
    """Test batch prediction"""
    ml_model = MLModel(sample_model)
    samples = [[0.5, -0.5, 1.0], [1.0, 0.0, -1.0]]
    
    results = ml_model.batch_predict(samples)
    
    assert len(results) == 2
    assert all('prediction' in result for result in results)

def test_invalid_input(sample_model):
    """Test error handling for invalid input"""
    ml_model = MLModel(sample_model)
    
    with pytest.raises(Exception):
        ml_model.predict([0.5])  # Too few features
    
    with pytest.raises(Exception):
        ml_model.predict([0.5, 0.5, 0.5, 0.5])  # Too many features