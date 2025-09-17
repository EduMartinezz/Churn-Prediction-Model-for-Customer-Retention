# 📉 Customer Churn Prediction — Predictor & Evaluator

A machine learning project that predicts **customer churn** using telecom customer data.  
This project demonstrates a **full ML workflow**: preprocessing, feature engineering, model training, threshold tuning, evaluation, and deployment with Streamlit.

---

## 📖 Project Overview
- Builds a predictive model for **Churn (Yes/No)** using demographic, service usage, and billing features.  
- Includes a **Streamlit dashboard** for:
  - Interactive evaluation (metrics, ROC/PR curves, confusion matrix)  
  - Prediction on new uploaded CSVs  
  - Downloadable predictions for further analysis  
- Demonstrates **portfolio-ready data science skills**:
  - Feature engineering (`tenure` binning, contract/payment risk flags)  
  - Pipeline-based preprocessing (no leakage)  
  - Model selection & threshold tuning  
  - Deployment for interactive exploration  

---

## 📊 Dataset
- **Source:** [Telco Customer Churn Dataset (IBM Sample)](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Target column:** `Churn` (`Yes` / `No`)  

**Features:**
- **Numeric (4):** `MonthlyCharges`, `SeniorCitizen`, `TotalCharges`, `tenure`  
- **Categorical (15):** `gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod`  
- **Dropped:** `customerID` (identifier, no predictive value)  

**Feature Engineering:**
- Collapsed `"No phone service"` → `"No"` in `MultipleLines`  
- Derived flags:  
  - `is_fiber`, `is_monthly`, `is_echeck`  
  - Interactions: `monthly_echeck`, `fiber_senior`  
- **Tenure binning** using `KBinsDiscretizer` (6 quantile bins, one-hot encoded)  
- `avg_charge = TotalCharges / tenure`  

---

## 🚀 Features
- **Leak-free Preprocessing**
  - Imputation (median for numerics, most-frequent for categoricals)  
  - Scaling (StandardScaler for numeric features)  
  - One-hot encoding of categoricals  
  - Tenure discretization into balanced bins  

- **Modeling**
  - Random Forest Classifier (balanced)  
  - HistGradientBoostingClassifier (boosted trees)  
  - Probability calibration for better thresholding  
  - Cross-validation for model selection  

- **Evaluation Dashboard**
  - Accuracy, F1, ROC AUC, PR AUC  
  - Confusion Matrix  
  - ROC Curve & Precision–Recall Curve  
  - Suggested **optimal decision threshold** (max F1)  
  - Top feature importance  

- **Prediction Dashboard**
  - Upload CSVs with the same schema  
  - Get predicted labels + probability of churn  
  - Download results as `predictions.csv`  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib  
- **Deployment:** Streamlit + ngrok (for Colab public URL)  
- **Environment:** Google Colab  

---

## 📈 Results

### Test Set Performance (default threshold = 0.50)
- Accuracy: **0.803**  
- F1 Score: **0.554**  
- ROC AUC: **0.834**  
- PR AUC: **0.642**

### Threshold Tuning (maximizing F1)
- Suggested Threshold: **0.30**  
- Accuracy: **0.760**  
- F1 Score: **0.633**  
- Precision (Churn class): **0.53**  
- Recall (Churn class): **0.79**  
- Weighted Avg F1: **0.77**

> **Business takeaway:** Lowering the threshold improves recall of churners significantly (79% vs ~55%), which is critical for retention campaigns.  
> In churn prediction, it’s usually better to **catch more at-risk customers**, even if some non-churners are flagged incorrectly.

---

## 📷 Screenshots

### Evaluation Dashboard
https://github.com/EduMartinezz/Churn-Prediction-Model-for-Customer-Retention/blob/main/appIntro.PNG
### Prediction Output
https://github.com/EduMartinezz/Churn-Prediction-Model-for-Customer-Retention/blob/main/app_predict.PNG
https://github.com/EduMartinezz/Churn-Prediction-Model-for-Customer-Retention/blob/main/app_predict2.PNG
---

## ▶️ Quickstart (Colab / Local)

1. Clone this repo or upload it to Colab:
   ```bash
   git clone https://github.com/your-username/churn-prediction.git
   cd churn-prediction

2. ## Install requirements:
    pip install -r requirements.txt

3. ## Train (or retrain) and export the model:
    import joblib
    joblib.dump(pipe, "churn_pipeline.joblib")

4. ## streamlit run app_churn.py
    streamlit run app_churn.py

5. ##  (Optional, in Colab) Expose public URL with ngrok:
    from pyngrok import ngrok
    import os
    os.environ["NGROK_TOKEN"] = "YOUR_TOKEN"
    ngrok.set_auth_token(os.environ["NGROK_TOKEN"])
    public_url = ngrok.connect(8501)
    print(public_url)



### 📂 Project Structure
churn-prediction/
├── app_churn.py             # Streamlit dashboard
├── churn_pipeline.joblib    # Saved pipeline model
├── Churn_Prediction_Project.ipynb   # Notebook (training + evaluation)
├── requirements.txt
├── README.md
└── docs/
    ├── eval_example.png
    └── pred_example.png


### 🔮 Future Improvements

Add SHAP explanations for interpretability

Try cost-sensitive loss (heavier penalty on missed churners)

Deploy on HuggingFace Spaces / Streamlit Cloud for persistent demo

