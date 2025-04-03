## Churn-Prediction-Model-for-Customer-Retention
**Build a machine learning model to predict which customers are likely to churn and identify factors driving their departure**

## Telco Customer Churn Prediction Pipeline
![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)  
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Create and Clone the Repository](#step-1-create-and-clone-the-repository)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Download the Dataset](#step-3-download-the-dataset)
  - [Step 4: Preprocess the Data](#step-4-preprocess-the-data)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Run the Streamlit App](#step-6-run-the-streamlit-app)
- [Usage](#usage)
  - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
  - [Using the Streamlit App](#using-the-streamlit-app)
- [Cloud Deployment](#cloud-deployment)
  - [Option 1: Streamlit Community Cloud](#option-1-streamlit-community-cloud)
  - [Option 2: Heroku](#option-2-heroku)
- [Results](#results)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
  - [Sample Prediction](#sample-prediction)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
The **Telco Customer Churn Prediction Pipeline** is a machine learning project designed to predict customer churn in the telecommunications industry. Customer churn, or attrition, refers to customers discontinuing their subscriptions with a telecom provider. This project aims to:
- Build a predictive model to identify customers at risk of churning.
- Analyze key factors driving churn (e.g., contract type, tenure, monthly charges).
- Provide an interactive interface for real-time churn predictions using a Streamlit app.

The project uses the Telco Customer Churn dataset from Kaggle, which contains data on 7,043 customers with features like demographics, service usage, and account information. The pipeline includes data preprocessing, exploratory data analysis (EDA), model training with hyperparameter tuning, feature importance analysis using SHAP, and a user-friendly Streamlit app for predictions.

This project was developed in Python Google Colab,and can be run locally, or deployed on the cloud. It demonstrates skills in data preprocessing, machine learning, model explainability, and web app development, making it a valuable addition to a data science portfolio.

## Features
- **Churn Prediction**: Predict whether a customer will churn using a Random Forest model with an ROC-AUC score of ~0.85.
- **Feature Importance**: Identify key factors driving churn (e.g., contract type, tenure, monthly charges) using SHAP analysis.
- **Interactive Interface**: A Streamlit app allows users to input customer data, predict churn, and visualize feature importance.
- **Preprocessing Pipeline**: Handles missing values, encodes categorical variables, scales numerical features, and addresses class imbalance using SMOTE.
- **Prediction History**: Tracks past predictions in the Streamlit app for easy reference.
- **Visualizations**: Includes EDA plots (e.g., churn distribution, tenure vs. churn) and feature importance charts.

## Dataset
The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. Key details:
- **Rows**: 7,043 customers.
- **Columns**: 21 features, including:
  - **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - **Service Usage**: `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
  - **Account Info**: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
  - **Target Variable**: `Churn` (Yes/No).
- **File Name**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Size**: ~1 MB (small enough to include in the repository, but excluded via `.gitignore` to encourage users to download from Kaggle).

The dataset is not included in the repository due to best practices for data management. Instructions for downloading it are provided in the [Setup Instructions](#setup-instructions).

## Repository Structure
The repository is organized as follows:

churn-prediction-pipeline/
├── data/                     # Raw and processed datasets
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed_data.csv
├── notebooks/               # Jupyter/Colab notebooks for exploration
│   └── preprocessing.ipynb
├── src/                     # Python scripts
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── app.py              # Streamlit app
├── models/                  # Saved models and scalers
│   ├── rf_churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── docs/                    # Documentation and visualizations
│   ├── churn_visualizations.png
│   └── project_report.pdf   (optional)
├── requirements.txt         # List of dependencies
├── .gitignore               # Files to exclude
└── README.md                # Project description

## Requirements
To run this project, you’ll need the following:
- **Python**: 3.8 or higher.
- **Dependencies**: Listed in `requirements.txt`:

pandas==2.0.2
  numpy==1.24.3
  scikit-learn==1.2.2
  imblearn==0.10.1
  matplotlib==3.7.1
  seaborn==0.12.2
  shap==0.42.0
  streamlit==1.24.0
  joblib==1.3.1


## Setup Instructions
Follow these steps to set up and run the project on your local machine, in Google Colab, or on a cloud platform.

### Step 1: Create and Clone the Repository
1. **Create the Repository on GitHub**:
 - Log in to [GitHub](https://github.com) with your account.
 - Click the “+” icon in the top-right corner and select “New repository.”
 - Name the repository `churn-prediction-model-for customer-retention`.
 - Set visibility to **Public**.
 - 
 - Click “Create repository.” Your repository will be available at `https://github.com/your-username/churn-prediction-pipeline`.

2. **Clone the Repository Locally (Optional)**:
 If you prefer working locally (e.g., with Jupyter Lab or VS Code), clone the repository:
 - Open a terminal (e.g., Anaconda Prompt, Command Prompt, or Bash).
 - Navigate to your desired project directory:
   ```bash
   cd /path/to/your/projects
   ```
 - Clone the repository:
   ```bash
   git clone https://github.com/your-username/churn-prediction-pipeline.git
   cd churn-prediction-pipeline
   ```
 - Alternatively, you can work directly in the GitHub web interface or Google Colab by uploading files manually.

### Step 2: Install Dependencies
1. **Create a Virtual Environment (Recommended)**:
 - Create a virtual environment to isolate dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
 - You’ll see `(venv)` in your terminal, indicating the virtual environment is active.

2. **Install Dependencies**:
 - Install the required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
 - If you encounter issues, ensure `pip` is up-to-date:
   ```bash
   pip install --upgrade pip
   ```

### Step 3: Download the Dataset
1. **Download the Telco Customer Churn Dataset**:
 - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
 - Save the file as `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` directory:
   ```
   churn-prediction-pipeline/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```
 - Alternatively, you can modify `src/preprocess_data.py` to download the dataset programmatically using `kaggle` API:
   ```bash
   kaggle datasets download -d blastchar/telco-customer-churn -p data/
   unzip data/telco-customer-churn.zip -d data/
   ```

### Step 4: Preprocess the Data
1. **Run the Preprocessing Script**:
 - The `src/preprocess_data.py` script handles data preprocessing, including handling missing values, encoding categorical variables, and scaling numerical features.
 - Run the script:
   ```bash
   python src/preprocess_data.py
   ```
 - This will generate `data/processed_data.csv` and save the scaler as `models/scaler.pkl`.

2. **Explore the Preprocessing Notebook (Optional)**:
 - Open `notebooks/preprocessing.ipynb` in Jupyter Notebook or Google Colab to explore the preprocessing steps and visualizations:
   ```bash
   jupyter notebook notebooks/preprocessing.ipynb
   ```
 - The notebook includes markdown cells explaining each step (e.g., “Load Dataset,” “Handle Missing Values,” “Visualizations”).

### Step 5: Train the Model
1. **Run the Training Script**:
 - The `src/train_model.py` script trains a Random Forest model with hyperparameter tuning using GridSearchCV.
 - Run the script:
   ```bash
   python src/train_model.py
   ```
 - This will generate the following files in the `models/` directory:
   - `rf_churn_model.pkl`: Trained Random Forest model.
   - `feature_names.pkl`: List of feature names for prediction.

### Step 6: Run the Streamlit App
1. **Launch the Streamlit App**:
 - The `src/app.py` script contains the Streamlit app for interactive churn predictions.
 - Run the app:
   ```bash
   streamlit run src/app.py
   ```
 - Open the provided URL (e.g., `http://localhost:8501`) in your browser to access the app.

2. **Test the App**:
 - Input customer details (e.g., tenure, monthly charges, contract type) and click “Predict Churn” to see the prediction and probability.
 - View feature importance and prediction history within the app.

**Note for Google Colab Users**:
- If you’re working in Google Colab, you can run the entire pipeline in a notebook:
1. Upload all files to Colab (or mount Google Drive).
2. Install dependencies in a Colab cell:
   ```bash
   !pip install -r requirements.txt
   ```
3. Run preprocessing and training scripts using `!python` commands.
4. For the Streamlit app, use `ngrok` to create a public URL (as shown in the previous project):
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("your-ngrok-authtoken")
   public_url = ngrok.connect(8501)
   print("Streamlit app is live at:", public_url)
   !streamlit run src/app.py --server.port 8501
   ```

## Usage
### Using the Jupyter Notebook
1. **Exploratory Data Analysis (EDA)**:
 - Open `notebooks/preprocessing.ipynb` to explore the dataset.
 - The notebook includes:
   - Data loading and cleaning.
   - Visualizations (e.g., churn distribution, tenure vs. churn, correlation heatmap).
   - Preprocessing steps (e.g., handling missing values, encoding, scaling).

2. **Key Visualizations**:
 - Churn distribution: Shows the imbalance between churned (26.5%) and non-churned (73.5%) customers.
 - Tenure vs. Churn: Boxplot comparing tenure for churned vs. non-churned customers.
 - Correlation Heatmap: Identifies relationships between features and churn.

### Using the Streamlit App
1. **Launch the App**:
 - Run `src/app.py` as described in the setup instructions.
 - Access the app in your browser (e.g., `http://localhost:8501`).

2. **Input Customer Details**:
 - Use the form to input customer data, including:
   - **Tenure**: Number of months the customer has been with the company (0–72).
   - **Monthly Charges**: Monthly bill amount ($0–$200).
   - **Total Charges**: Total amount billed ($0–$10,000).
   - **Contract Type**: Month-to-month, One year, or Two year.
   - **Internet Service**: DSL, Fiber optic, or No.
   - **Payment Method**: Electronic check, Mailed check, Bank transfer, or Credit card.
   - **Paperless Billing**: Yes or No.
   - **Tech Support**: Yes, No, or No internet service.
   - **Online Security**: Yes, No, or No internet service.

3. **Predict Churn**:
 - Click “Predict Churn” to see the prediction (Yes/No) and churn probability (e.g., 65%).
 - View the top 10 features driving the prediction using a bar plot.
 - Check the prediction history table to review past predictions.

4. **Example Input**:
 - **Tenure**: 12 months
 - **Monthly Charges**: $80
 - **Total Charges**: $960
 - **Contract**: Month-to-month
 - **Internet Service**: Fiber optic
 - **Payment Method**: Electronic check
 - **Paperless Billing**: Yes
 - **Tech Support**: No
 - **Online Security**: No
 - **Output**:
   - Churn Prediction: Yes
   - Churn Probability: 65%

## Cloud Deployment(optional)
### Option 1: Streamlit Community Cloud (Recommended for Prototyping)
1. **Push to GitHub**:
 - Ensure all files are committed and pushed to your repository:
   ```bash
   git add .
   git commit -m "Add complete churn prediction pipeline"
   git push origin main
   ```

2. **Deploy on Streamlit Community Cloud**:
 - Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud).
 - Connect your GitHub account and select the `churn-prediction-pipeline` repository.
 - Specify the main script as `src/app.py`.
 - Deploy the app.
 - **Note**: The model files (`rf_churn_model.pkl`, `scaler.pkl`, `feature_names.pkl`) must be included in the repository or hosted on a cloud storage service (e.g., Google Drive) and downloaded at runtime. Modify `app.py` to download these files if needed:
   ```python
   import gdown
   gdown.download("your-google-drive-link-to-rf_churn_model.pkl", "models/rf_churn_model.pkl", quiet=False)
   ```

3. **Access the App**:
 - Once deployed, Streamlit Community Cloud will provide a public URL (e.g., `https://your-app.streamlit.app`).
 - Update the README with the URL:
   ```markdown
   [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
   ```

### Option 2: Heroku (Recommended for Small Team Usage)
1. **Prepare for Heroku**:
 - Create a `Procfile` in the root directory:
   ```
   web: streamlit run src/app.py --server.port $PORT
   ```
 - Ensure `requirements.txt` is up-to-date.

2. **Deploy on Heroku**:
 - Install the Heroku CLI and log in:
   ```bash
   heroku login
   ```
 - Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```
 - Deploy the app:
   ```bash
   git push heroku main
   ```
 - Open the app:
   ```bash
   heroku open
   ```
 - **Note**: Host the model files on a cloud storage service and download them at runtime, as Heroku’s storage is limited.

## Results
### Model Performance
- **Random Forest Model**:
- **Accuracy**: ~80%
- **Precision (Churn)**: ~55%
- **Recall (Churn)**: ~70%
- **ROC-AUC**: ~0.85
- The model effectively identifies customers at risk of churning, with a high recall for the churn class, ensuring most at-risk customers are flagged.

### Key Insights
- **Churn Rate**: 26.5% of customers churned.
- **Top Features Driving Churn** (from SHAP analysis):
- `Contract_Month-to-month`: Customers on month-to-month contracts are 42.7% more likely to churn compared to those on one-year or two-year contracts.
- `tenure`: Lower tenure (e.g., <12 months) is strongly associated with churn.
- `MonthlyCharges`: Higher monthly charges (e.g., >$70) correlate with higher churn rates.
- `InternetService_Fiber optic`: Fiber optic users have a higher churn rate (41.9%) compared to DSL (18.9%) or no internet (7.4%).
- **Demographic Insights**:
- Senior citizens have a higher churn rate (41.7%) compared to non-senior citizens (23.6%).
- Customers without tech support or online security are more likely to churn.

### Sample Prediction
- **Input**:
- Tenure: 12 months
- Monthly Charges: $80
- Total Charges: $960
- Contract: Month-to-month
- Internet Service: Fiber optic
- Payment Method: Electronic check
- Paperless Billing: Yes
- Tech Support: No
- Online Security: No
- **Output**:
- Churn Prediction: Yes
- Churn Probability: 65%
- **Interpretation**: The customer is at high risk of churning, likely due to the month-to-month contract, high monthly charges, and lack of tech support or online security.

## Visualizations
The project includes several visualizations to provide insights into the data and model:

1. **Churn Distribution**:
 - Shows the imbalance between churned (26.5%) and non-churned (73.5%) customers.
 - Generated in `notebooks/preprocessing.ipynb` and saved as `docs/churn_visualizations.png`.

2. **Tenure vs. Churn**:
 - Boxplot comparing tenure for churned vs. non-churned customers, highlighting that churned customers have lower tenure (median: ~10 months) compared to non-churned (median: ~37 months).

3. **Feature Importance**:
 - Bar plot of the top 10 features driving churn predictions, generated in the Streamlit app.
 - Example: `Contract_Month-to-month` and `tenure` are the most influential features.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
 ```bash
 git checkout -b feature/your-feature-name

**Make your changes and commit:
git commit -m "Add your feature"

**Push to your branch:
git push origin feature/your-feature-name
