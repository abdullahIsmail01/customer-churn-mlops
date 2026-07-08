# Customer Churn Prediction using MLOps

## Project Overview

This project implements a complete MLOps pipeline for customer churn prediction using the IBM Telco Customer Churn dataset. The objective is to build a reproducible machine learning workflow including data preprocessing, feature engineering, model training, evaluation, experiment tracking with MLflow, and deployment using Docker.

---

## Dataset

The project uses the **IBM Telco Customer Churn** dataset containing approximately 7,000 customer records.

**Target Variable**

* **Churn Value = 1** → Customer left the company.
* **Churn Value = 0** → Customer stayed with the company.

---

## Data Versioning

The dataset is managed in three versions.

### Version 1 (v1)

* Original raw dataset.

### Version 2 (v2)

* Removed unnecessary columns.
* Converted categorical features using one-hot encoding.
* Handled missing values.
* Converted numerical columns to appropriate data types.

### Version 3 (v3)

Additional features were generated, including:

* Has Family
* Protection Services Count
* Streaming Services Count
* Average Monthly Spending
* Is Long-Term Customer

---

## Machine Learning Models

The following classification algorithms were implemented and compared:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

---

## Model Evaluation

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

To improve reliability, the training process includes:

* Stratified Train/Test Split
* 5-Fold Cross Validation
* Fixed Random Seed (42)

The model with the highest Cross Validation Accuracy is selected as the final model and evaluated on the test dataset.

---

## MLflow Tracking

MLflow is used to record every experiment, including:

* Model parameters
* Evaluation metrics
* Dataset version
* Random seed
* Trained model
* Confusion Matrix

---

## Docker Deployment

The complete pipeline is containerized using Docker.

The Docker image automatically performs:

1. Load dataset
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model selection
6. Model evaluation
7. MLflow logging

---

## Project Structure

project/
│
├── data/
│   ├── v1/
│   │   └── Telco_customer_churn.xlsx
│   ├── v2/
│   │   └── Telco_customer_churn_preprocessed.xlsx
│   └── v3/
│       └── Telco_customer_churn_features.xlsx
│
├── mlruns/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_utils.py
│
├── Dockerfile
├── mlflow.db
├── requirements.txt
├── run_pipeline.py
├── README.md
├── .gitignore
└── .dockerignore
---

## Installation

Clone the repository:

```bash
git clone https://github.com/abdullahIsmail01/customer-churn-mlops.git
cd customer-churn-mlops
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run the complete MLOps pipeline:

```bash
python run_pipeline.py
```

---

## Running with Docker

Build the Docker image:

```bash
docker build -t customer-churn-mlops .
```

Run the container:

```bash
docker run --rm customer-churn-mlops
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* MLflow
* Docker
* Matplotlib
* Seaborn

---

## Author

Abdullah Ismail

Software Engineering Student
