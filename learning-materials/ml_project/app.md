Awesome 🚀 — now you’ve put everything together into a **FastAPI web service** for your ML model! Let’s go through `app.py` step by step 👇

---

## **1. Imports**

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.model import MLModel
from src.schemas import (
    PredictionInput, PredictionOutput, 
    BatchPredictionInput, BatchPredictionOutput,
    HealthResponse
)
import os
```

* **FastAPI** → framework for building APIs.
* **HTTPException** → to return proper error codes.
* **CORS middleware** → allows frontend apps (like React, Vue, etc.) to call your API.
* **MLModel** → your wrapper around the trained model (`model.py`).
* **Schemas** → request/response validation (`schemas.py`).

---

## **2. FastAPI App Setup**

```python
app = FastAPI(title="ML Model API", version="1.0.0")
```

* Initializes FastAPI app.
* Adds metadata (title + version).

---

## **3. CORS Middleware**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

* This makes your API accessible from **any frontend client** (browser, React app, etc.).
* `allow_origins=["*"]` means no restrictions → good for testing, but in production you’d restrict to your domain.

---

## **4. Model Loading**

```python
try:
    model = MLModel("model.pkl")
    model_loaded = True
except Exception as e:
    model_loaded = False
    print(f"Failed to load model: {e}")
```

* Tries to load the trained model (`model.pkl`).
* Sets `model_loaded = True` if successful, otherwise logs an error.
* This avoids crashing if the model file is missing or corrupted.

---

## **5. Health Check Endpoint**

```python
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }
```

* Endpoint: `GET /`
* Returns service status and whether the model is loaded.
* Example:

  ```json
  { "status": "healthy", "model_loaded": true }
  ```

---

## **6. Single Prediction Endpoint**

```python
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Single prediction endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = model.predict(input_data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

* Endpoint: `POST /predict`
* Input: `PredictionInput` → exactly 3 features.
* Calls `model.predict()` → returns class, probabilities, and confidence.
* Example request:

  ```json
  { "features": [0.5, -1.2, 3.0] }
  ```
* Example response:

  ```json
  {
    "prediction": 1,
    "probabilities": [0.35, 0.65],
    "confidence": 0.65
  }
  ```

---

## **7. Batch Prediction Endpoint**

```python
@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Batch prediction endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = model.batch_predict(input_data.samples)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

* Endpoint: `POST /predict_batch`
* Input: multiple samples (`BatchPredictionInput`).
* Calls `model.batch_predict()` → returns predictions for each sample.
* Example request:

  ```json
  {
    "samples": [
      [0.1, 0.5, -0.3],
      [1.2, -0.7, 0.8]
    ]
  }
  ```
* Example response:

  ```json
  {
    "predictions": [
      {"prediction": 0, "probabilities": [0.6, 0.4], "confidence": 0.6},
      {"prediction": 1, "probabilities": [0.2, 0.8], "confidence": 0.8}
    ]
  }
  ```

---

## **8. Run the App**

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

* Starts the FastAPI server using **Uvicorn**.
* Runs at: [http://localhost:8000](http://localhost:8000)

---

## **9. Auto Docs**

With FastAPI you automatically get:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* Redoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

Both will use your **schemas** to show input/output formats.

---

## ✅ Summary

This `app.py` is a **production-ready ML API** with:

* `/` → health check
* `/predict` → single prediction
* `/predict_batch` → multiple predictions
* **CORS enabled** so frontend apps can use it
* **automatic validation** thanks to Pydantic schemas

---

Would you like me to also show you how to **Dockerize this FastAPI app** so you can run it anywhere with:

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```
