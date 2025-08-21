# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app import app
from src.model import MLModel
import numpy as np

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'model_loaded' in data

def test_predict_endpoint(client, mocker):
    """Test prediction endpoint"""
    # Mock the model prediction
    mock_prediction = {
        'prediction': 1,
        'probabilities': [0.3, 0.7],
        'confidence': 0.7
    }
    
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = mock_prediction
    mocker.patch('app.model', mock_model)
    mocker.patch('app.model_loaded', True)
    
    # Test valid request
    response = client.post("/predict", json={"features": [0.5, -0.5, 1.0]})
    assert response.status_code == 200
    data = response.json()
    assert data == mock_prediction
    
    # Test invalid request
    response = client.post("/predict", json={"features": [0.5]})
    assert response.status_code == 400

def test_batch_predict_endpoint(client, mocker):
    """Test batch prediction endpoint"""
    mock_predictions = [
        {
            'prediction': 1,
            'probabilities': [0.3, 0.7],
            'confidence': 0.7
        },
        {
            'prediction': 0,
            'probabilities': [0.6, 0.4],
            'confidence': 0.6
        }
    ]
    
    mock_model = mocker.MagicMock()
    mock_model.batch_predict.return_value = mock_predictions
    mocker.patch('app.model', mock_model)
    mocker.patch('app.model_loaded', True)
    
    response = client.post("/predict_batch", json={
        "samples": [[0.5, -0.5, 1.0], [1.0, 0.0, -1.0]]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 2