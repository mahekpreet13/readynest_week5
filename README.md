# End-to-End Data Pipeline & Predictive Analytics

This repository contains a complete e-commerce analytics project built for the **ReadyNest Week 5 Task**. It includes synthetic data generation, ETL, exploratory analysis, a lightweight prediction model, and an interactive Streamlit dashboard.

## Overview

The project follows a simple analytics pipeline:

1. Generate a realistic synthetic sales dataset.
2. Clean and transform the raw data.
3. Produce EDA charts and business summaries.
4. Train a regression model to predict order revenue.
5. Serve everything through a Streamlit dashboard.

## Features

- **Data Collection**: Generates a realistic synthetic e-commerce dataset with 5,000+ rows.
- **ETL Pipeline**: Cleans data, handles duplicates and missing values, and engineers time-based and anomaly features.
- **Exploratory Data Analysis**: Produces charts for revenue trends, category performance, city distribution, discount impact, weekday patterns, and more.
- **Predictive Modeling**: Trains a lightweight regression model to predict order revenue and evaluates it with RMSE, MAE, and R².
- **Streamlit Dashboard**: Provides KPI cards, interactive charts, an ML prediction form, and a business insights page.

## Project Structure

```text
task4/
├── README.md
├── requirements.txt
├── src/
│   ├── data_collection.py    # Synthetic data generation
│   ├── etl_pipeline.py       # Clean, transform, and load pipeline
│   ├── eda.py                # EDA charts and summaries
│   └── model.py              # Revenue prediction model
├── dashboard/
│   ├── app.py                # Streamlit app
│   └── *.png / *.jpg         # Dashboard background assets
├── data/
│   ├── raw/                  # Generated raw dataset
│   └── cleaned/              # Cleaned CSV outputs
├── models/                   # Saved model artifact
└── reports/
    ├── charts/               # Generated visualization outputs
    └── business_insights.md  # Final business recommendations
```

## Tech Stack

- **Language**: Python
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Dashboard**: `streamlit`
- **Machine Learning**: custom ridge-style regression, `scikit-learn`, `scipy`, `statsmodels`, `xgboost`
- **Utilities**: `pickle`, `joblib`, `openpyxl`, `faker`

## Setup

1. Clone the repository or open the project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

Run the scripts in order:

```bash
python src/data_collection.py
python src/etl_pipeline.py
python src/eda.py
python src/model.py
streamlit run dashboard/app.py
```

## Important Note

The current codebase has a small file-name mismatch to be aware of:

- `src/data_collection.py` writes the raw file to `data/raw/ecommerce_sales.csv`
- `src/etl_pipeline.py` expects the raw file at `data/ecommerce_sales_jharkhand.csv`

If you run the pipeline locally, either:

- update `RAW_PATH` in `src/etl_pipeline.py` to point to `data/raw/ecommerce_sales.csv`, or
- copy/rename the generated raw file to the path expected by the ETL script.

## Deploy on Streamlit via GitHub

You can deploy this app on **Streamlit Community Cloud** using GitHub:

1. Push the project to a GitHub repository.
2. Make sure the repo includes the files the dashboard needs at runtime:
   - `data/cleaned/ecommerce_cleaned.csv`
   - `data/cleaned/monthly_summary.csv`
   - `models/revenue_model.pkl`
3. Open Streamlit Community Cloud and create a new app from your GitHub repo.
4. Set the app entry point to:

```text
dashboard/app.py
```

5. Deploy.

### Deployment Tip

Streamlit Cloud will not run your local pipeline automatically. For the app to work on first deploy, commit the generated cleaned data and model files, or update the app to regenerate them if they are missing.

If Streamlit Cloud builds your app with a newer Python version and dependency compilation fails, change the app's Python version in Streamlit Community Cloud advanced settings and redeploy. The docs note that Community Cloud defaults to Python 3.12 and lets you select the version during deployment. See:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Outputs

- Raw generated data in `data/raw/`
- Cleaned data in `data/cleaned/`
- EDA charts in `reports/charts/`
- Saved ML model in `models/`
- Business insights in `reports/business_insights.md`

## Evaluation Coverage

- **Data Collection**: synthetic dataset generation
- **Data Cleaning & ETL**: null handling, duplicates, and feature engineering
- **EDA**: automated statistical plots and summaries
- **ML Model**: regression-based revenue prediction and evaluation
- **Dashboard**: interactive Streamlit app with multiple pages
- **Insights**: business recommendations based on analysis
