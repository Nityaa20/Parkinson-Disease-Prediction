# 🧠 Parkinson's Disease AI Predictor

> An AI-powered web application that predicts Parkinson's Disease from vocal biomarker measurements using Machine Learning.

---

## 📌 Overview

Parkinson's Disease is one of the most common neurodegenerative disorders, and **early detection is critical** for better disease management. Vocal changes are among the **earliest and most measurable symptoms** of Parkinson's — but manual analysis of vocal biomarkers is not feasible at scale.

This application solves that problem by allowing users to input **22 vocal biomarker features** and receive an **instant AI-powered prediction** along with a confidence score and the top contributing factors.

---

## ✨ Features

- 🎯 **Instant Prediction** — Input 22 vocal features and get real-time Parkinson's detection results
- 📊 **Confidence Score** — Displays the model's confidence percentage for each prediction
- 🔍 **Feature Importance** — Shows the top vocal factors that influenced the prediction
- ⚡ **Fast API Backend** — Built with FastAPI for high-performance inference
- 🔒 **Pre-trained Model** — Uses a trained Random Forest classifier loaded via joblib
- 📐 **Scaled Inputs** — StandardScaler ensures all inputs are normalized before inference

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🐍 Language | Python 3.x |
| 🚀 API Framework | FastAPI |
| 🤖 ML Model | Random Forest Classifier (scikit-learn) |
| 📦 Model Loading | joblib |
| 🔢 Numerical Computing | NumPy |
| ⚙️ Input Scaling | StandardScaler (scikit-learn) |
| 🌐 Server | Uvicorn (ASGI) |

---

## 📁 Project Structure

```
backend/
│
├── 📄 main.py                  # FastAPI application entry point
├── 📄 model.pkl                # Pre-trained Random Forest model
├── 📄 scaler.pkl               # Saved StandardScaler for input normalization

│
├── 📂 frontend/
│   ├── index.html               # Model training script
│                
│
├── 📂 data/
│   └── parkinsons.csv          # UCI Parkinson's dataset
│
└── 📄 README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/parkinsons-predictor.git
cd parkinsons-predictor
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
uvicorn main:app --reload
```

### 4️⃣ Access the API

```
http://localhost:8000
http://localhost:8000/docs   ← Interactive Swagger UI
```

---

## 📦 Dependencies

```txt
fastapi
uvicorn
scikit-learn
numpy
joblib
```

Install all at once:

```bash
pip install fastapi uvicorn scikit-learn numpy joblib
```

---

## 🔗 API Endpoints

### `POST /predict`

Accepts 22 vocal biomarker features and returns a prediction.

**Request Body:**
```json
{
  "MDVP_Fo": 119.992,
  "MDVP_Fhi": 157.302,
  "MDVP_Flo": 74.997,
  "MDVP_Jitter": 0.00784,
  "MDVP_Shimmer": 0.04374,
  "NHR": 0.02211,
  "HNR": 21.033,
  "RPDE": 0.414783,
  "DFA": 0.815285,
  "spread1": -4.813031,
  "spread2": 0.266482,
  "D2": 2.301442,
  "PPE": 0.284654
  ...
}
```

**Response:**
```json
{
  "prediction": "Parkinson's Detected",
  "confidence": 94.3,
  "top_features": [
    {"feature": "PPE", "importance": 0.187},
    {"feature": "spread1", "importance": 0.154},
    {"feature": "RPDE", "importance": 0.132}
  ]
}
```

---

## 🧪 How It Works

```
User Input (22 vocal features)
        ↓
  StandardScaler (Normalization)
        ↓
  Random Forest Model (Inference)
        ↓
  NumPy (Feature Importance + Probabilities)
        ↓
  FastAPI Response (Prediction + Confidence Score)
```

1. **User inputs** 22 vocal biomarker values through the API
2. Inputs are **normalized** using the saved `StandardScaler`
3. The **pre-trained Random Forest model** performs inference
4. **NumPy** computes class probabilities and feature importances
5. Results are returned with **prediction label + confidence %**

---

## 📊 Dataset

This model was trained on the **UCI Parkinson's Disease Dataset**.

| Property | Value |
|---|---|
| 📋 Samples | 195 voice recordings |
| 🔢 Features | 22 vocal biomarkers |
| 🎯 Target | Health status (1 = Parkinson's, 0 = Healthy) |
| 📌 Source | UCI Machine Learning Repository |

### Key Vocal Features Used:

- **MDVP:Fo(Hz)** — Average vocal fundamental frequency
- **MDVP:Jitter(%)** — Variation in fundamental frequency
- **MDVP:Shimmer** — Variation in amplitude
- **NHR / HNR** — Noise-to-harmonics ratio
- **RPDE** — Recurrence period density entropy
- **DFA** — Signal fractal scaling exponent
- **PPE** — Pitch period entropy

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| ✅ Accuracy | ~92% |
| 🎯 Precision | ~93% |
| 📈 Recall | ~96% |
| 📊 F1 Score | ~94% |

> Model trained using an 80/20 train-test split with cross-validation.

---

## 🚀 Future Improvements

- [ ] 🎤 Add direct audio file upload for automatic feature extraction
- [ ] 📱 Build a mobile-friendly frontend interface
- [ ] 🔄 Retrain with larger, more diverse datasets
- [ ] 📉 Add SHAP visualizations for better model explainability
- [ ] ☁️ Deploy to cloud (AWS / Render / Railway)

---

## 📄 License

This project is intended for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

> ⚠️ **Disclaimer:** This tool is an AI-based predictor and should not be used as a replacement for clinical diagnosis. Always consult a qualified medical professional.
