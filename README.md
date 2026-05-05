# 💊 MediRec AI — Medicine Recommendation System

> **Final Year Project** | Machine Learning & Deep Learning | Python + Streamlit

---

## 📌 Project Overview

MediRec AI predicts the most probable disease from a patient's symptoms and profile, then recommends ranked medicines with dosage and safety information. It trains and compares **7 models** (5 ML + 2 DL) and serves results through a polished Streamlit web app.

---

## 🗂️ Directory Structure

```
medicine_recommendation_system/
├── dataset/
│   ├── generate_dataset.py   ← Run first to generate CSV files
│   ├── symptoms_diseases.csv ← Auto-generated training data
│   └── medicine_data.csv     ← Medicine → disease mapping
├── models/                   ← Saved models & artefacts (auto-created)
├── assets/                   ← Evaluation plots (auto-created)
├── notebooks/
│   └── training.ipynb        ← Google Colab training notebook
├── app.py                    ← Streamlit web application
├── recommendation.py         ← Recommendation engine
├── preprocess.py             ← Data cleaning & feature engineering
├── train.py                  ← Model training + hyperparameter tuning
├── evaluate.py               ← Evaluation metrics & visualisations
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11
- pip
- (Optional) CUDA-compatible GPU for faster DL training

### Step 1 — Clone / Download the project

```bash
cd medicine_recommendation_system
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project (in order)

### 1️⃣ Generate Datasets

```bash
python dataset/generate_dataset.py
```

Creates:
- `dataset/symptoms_diseases.csv` — 3,000 training samples across 20 diseases
- `dataset/medicine_data.csv` — 60+ medicine entries with dosage & safety

---

### 2️⃣ Pre-process Data

```bash
python preprocess.py
```

Saves to `models/`:
- `label_encoder.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

---

### 3️⃣ Train All Models

```bash
python train.py
```

Trains:
| Model | Type |
|---|---|
| Random Forest | ML |
| SVM | ML |
| K-Nearest Neighbors | ML |
| Naive Bayes | ML |
| Decision Tree | ML |
| ANN (Keras) | DL |
| LSTM (Keras) | DL |

Saves to `models/`:
- `best_model.pkl` (best ML model by F1-score)
- `ann_model.keras`
- `lstm_model.keras`
- `all_metrics.json`

> ⏱️ Training may take **10–30 minutes** depending on hardware.

---

### 4️⃣ Evaluate & Generate Plots

```bash
python evaluate.py
```

Generates in `assets/`:
- `model_comparison.png`
- `confusion_matrix.png`
- `roc_curves.png`
- `ann_history.png`

---

### 5️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📊 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, how it works, key stats |
| 🩺 Symptom Input | Patient form — symptoms, age, gender, history |
| 📋 Results | Predicted disease + ranked medicine cards |
| 📊 Model Performance | Accuracy tables, confusion matrix, ROC curves |
| ℹ️ About | Abstract, tools, project structure |

---

## 🧪 Diseases Covered

Flu, Common Cold, COVID-19, Pneumonia, Bronchitis, Diabetes, Hypertension, Migraine, Gastritis, Typhoid, Malaria, Dengue, Tuberculosis, Anemia, Asthma, Urinary Tract Infection, Arthritis, Depression, Anxiety, Hypothyroidism

---

## 📓 Google Colab Notebook

Open `notebooks/training.ipynb` in Google Colab for a step-by-step walkthrough of the entire pipeline with rich markdown explanations.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | ≥1.3 | ML models, preprocessing, evaluation |
| tensorflow | ≥2.13 | ANN & LSTM models |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| streamlit | ≥1.28 | Web UI |
| matplotlib / seaborn | — | Visualisation |
| joblib | — | Model serialisation |

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It does not replace professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional.

---

## 👨‍💻 Team

| Role | Contribution |
|---|---|
| ML Engineer | Model design, training, evaluation |
| Full-Stack Developer | Streamlit UI, recommendation engine |
| Data Engineer | Dataset creation, preprocessing pipeline |

---

*Built with ❤️ as a Final Year Project — 2025*
