Perfect — now you’ve built a **Streamlit frontend** for your FastAPI ML API 👌. Let’s go through `streamlit_app.py` step by step.

---

## **1. Imports**

```python
import streamlit as st
import numpy as np
import requests
import json
```

* **streamlit** → creates the UI.
* **requests** → makes HTTP requests to your FastAPI API.
* **json** → formats input/output data.
* **numpy** (not strictly used here, but handy for numeric ops).

---

## **2. Page Config**

```python
st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="wide"
)
```

* Sets up the Streamlit page:

  * Title = *ML Model Demo*
  * Icon = 🤖
  * Wide layout (more space).

---

## **3. Title & Description**

```python
st.title("🤖 Machine Learning Model Demo")
st.markdown("Interact with our trained ML model through this web interface")
```

* Adds a page title and description.

---

## **4. Sidebar (API Config)**

```python
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "API URL", 
    "http://localhost:8000"
)
```

* Sidebar allows user to enter API URL.
* Default points to local FastAPI server: `http://localhost:8000`.

---

## **5. Tabs Layout**

```python
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "API Health"])
```

* Creates **3 tabs**:

  * Single Prediction
  * Batch Prediction
  * API Health

---

## **6. Tab 1 → Single Prediction**

```python
with tab1:
    st.header("Single Prediction")
    
    col1, col2 = st.columns(2)
```

* Splits page into two columns:

  * **col1** → input sliders
  * **col2** → prediction results

### **Inputs**

```python
feature1 = st.slider("Feature 1", -3.0, 3.0, 0.0, 0.1)
feature2 = st.slider("Feature 2", -3.0, 3.0, 0.0, 0.1)
feature3 = st.slider("Feature 3", -3.0, 3.0, 0.0, 0.1)
```

* User selects feature values with sliders.

### **Send to API**

```python
if st.button("Predict", key="single_predict"):
    response = requests.post(
        f"{api_url}/predict",
        json={"features": [feature1, feature2, feature3]}
    )
```

* When clicked, sends `POST /predict` request to FastAPI.
* Input JSON matches your Pydantic schema.

### **Display Results**

```python
if response.status_code == 200:
    result = response.json()
    with col2:
        st.subheader("Prediction Results")
        st.metric("Prediction", result['prediction'])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        st.subheader("Class Probabilities")
        prob_data = {
            'Class 0': result['probabilities'][0],
            'Class 1': result['probabilities'][1]
        }
        st.bar_chart(prob_data)
```

* Shows:

  * Prediction result (class 0 or 1).
  * Confidence (as percentage).
  * Bar chart of class probabilities.

---

## **7. Tab 2 → Batch Prediction**

```python
with tab2:
    st.header("Batch Prediction")
    
    sample_data = [
        [0.5, -1.2, 0.8],
        [-0.3, 1.5, -0.7],
        [1.0, 0.5, -1.0]
    ]
```

* Provides default sample batch.

### **JSON Input Box**

```python
input_json = st.text_area(
    "Enter samples as JSON array",
    value=json.dumps(sample_data, indent=2),
    height=200
)
```

* User can edit/enter samples in raw JSON format.

### **Send to API**

```python
if st.button("Predict Batch", key="batch_predict"):
    samples = json.loads(input_json)
    response = requests.post(
        f"{api_url}/predict_batch",
        json={"samples": samples}
    )
```

* Sends `POST /predict_batch` with samples.

### **Display Table**

```python
if response.status_code == 200:
    results = response.json()
    results_data = []
    for i, pred in enumerate(results['predictions']):
        results_data.append({
            'Sample': i+1,
            'Prediction': pred['prediction'],
            'Confidence': f"{pred['confidence']:.2%}",
            'Prob Class 0': f"{pred['probabilities'][0]:.3f}",
            'Prob Class 1': f"{pred['probabilities'][1]:.3f}"
        })
    st.table(results_data)
```

* Shows a table with:

  * Sample number
  * Prediction
  * Confidence
  * Probabilities

---

## **8. Tab 3 → API Health Check**

```python
with tab3:
    st.header("API Health Check")
    
    if st.button("Check Health"):
        response = requests.get(f"{api_url}/")
```

* Calls `GET /` (health check).

### **Show Results**

```python
if response.status_code == 200:
    health = response.json()
    st.success("✅ API is reachable")
    st.json(health)
    if health['model_loaded']:
        st.success("✅ Model is loaded and ready")
    else:
        st.error("❌ Model is not loaded")
```

* Displays API health info:

  * Whether API is reachable
  * Whether model is loaded

---

## ✅ **Summary**

* This **Streamlit frontend** talks to your **FastAPI backend**.
* Features:

  * **Single Prediction** → sliders + probability chart.
  * **Batch Prediction** → enter multiple samples in JSON + see table.
  * **Health Check** → check if API + model are running.

---

👉 Workflow:

1. Run FastAPI backend:

   ```bash
   uvicorn app:app --reload
   ```
2. Run Streamlit frontend:

   ```bash
   streamlit run streamlit_app.py
   ```
3. Open browser → Streamlit UI → interact with model.

---

Would you like me to also **Dockerize both FastAPI + Streamlit** in one container (with separate ports), so you can deploy this as a single ML app?
