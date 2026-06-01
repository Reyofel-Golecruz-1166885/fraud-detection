# 💳 Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions using XGBoost, deployed as a REST API with an interactive Streamlit dashboard.

---

## 📊 Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9699 | 0.06 | 0.92 | 0.11 |
| XGBoost (tuned) | 0.9821 | 0.67 | 0.88 | 0.76 |

---

## 🗂️ Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   └── processed/              # Scaled and resampled data
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Scaling, splitting, SMOTE
│   ├── 03_training.ipynb       # Baseline and XGBoost trainin      
│   ├── 04_evaluation.ipynb     # Metrics and visualisations
│   └── 05_shap.ipynb           # Model explainability
├── models/
│   ├── fraud_model_tuned.joblib
│   └── scaler.joblib
├── api/
│   └── main.py                 # FastAPI inference endpoint
├── dashboard/
│   └── app.py                  # Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ Key Technical Decisions

**Class imbalance** — The dataset is severely imbalanced (0.17% fraud).
SMOTE was applied exclusively on the training set after the train/test
split to prevent data leakage, balancing the classes from 394 to 227,451
fraud samples.

**Data leakage prevention** — StandardScaler was fit only on training data
and used to transform the test set, ensuring no test information influenced
the model.

**Metric selection** — Accuracy was discarded as the primary metric due to
class imbalance. ROC-AUC, Precision, Recall, and F1 were used instead.

**Model explainability** — SHAP (SHapley Additive exPlanations) was used
to explain individual predictions, identifying V14 and V4 as the strongest
fraud indicators.

---

## 📈 Confusion Matrix (Tuned XGBoost)

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 56,822 ✅ | 42 ⚠️ |
| **Actual Fraud** | 12 ❌ | 86 ✅ |

- Caught **86 out of 98** real fraud cases
- Only **42 false alarms** out of 56,864 legit transactions
- Missed **12 fraud** cases

---

## 🚀 Running the Project

### 1. Clone and install
```bash
git clone https://github.com/Reyofe-Golecruz-1166885-fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in `data/raw/`.

### 3. Run notebooks in order
```
01_eda → 02_preprocessing → 03_training → 04_evaluation → 05_shap
```

### 4. Start the API
```bash
cd api
uvicorn main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

### 5. Start the dashboard
```bash
cd dashboard
streamlit run app.py
```
Dashboard available at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| scikit-learn | Logistic Regression, preprocessing, evaluation |
| XGBoost | Main classification model |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Model explainability |
| FastAPI | REST API endpoint |
| Streamlit | Interactive dashboard |
| pandas, numpy | Data manipulation |

---

## 📁 Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
by ULB Machine Learning Group. Contains 284,807 transactions with 492 fraud cases (0.17%).
Features V1-V28 are PCA-transformed to protect customer privacy.

