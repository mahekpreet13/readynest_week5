# End-to-End Data Pipeline & Predictive Analytics

This repository contains a complete data analytics solution built for the **ReadyNest Week 5 Task**. It covers everything from synthetic data generation and ETL to Machine Learning and an interactive dashboard.

## 🚀 Features

- **Data Collection**: Generates a realistic synthetic e-commerce sales dataset (5,000+ rows).
- **ETL Pipeline**: Cleans data, engineers advanced features (date parts, profit margins, anomalies), and exports cleaned data.
- **Exploratory Data Analysis (EDA)**: Produces comprehensive charts analyzing revenue trends, customer segments, channel performance, and discount impacts.
- **Predictive ML Model**: Uses a lightweight regression model to predict order revenue based on multiple features, evaluated with RMSE and R².
- **Interactive Dashboard**: A **Streamlit** app featuring KPIs, interactive charts, and a real-time ML prediction interface.

## 📁 Project Structure

```text
task4/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw collected data (generated)
│   └── cleaned/              # Post-ETL cleaned data
├── src/
│   ├── data_collection.py    # Generates synthetic data
│   ├── etl_pipeline.py       # Transform & Load pipeline
│   ├── eda.py                # EDA utilities & charts
│   └── model.py              # ML model (lightweight regression)
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
├── models/                   # Saved ML models (.pkl)
└── reports/
    ├── charts/               # EDA visualization outputs
    └── business_insights.md  # Final insights & recommendations
```

## 🛠️ Setup & Installation

1. **Clone the repository** (or navigate to this directory)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 Running the Pipeline

Run the pipeline sequentially to generate data, process it, train the model, and launch the dashboard:

1. **Collect Data**:
   ```bash
   python src/data_collection.py
   ```
2. **Run ETL**:
   ```bash
   python src/etl_pipeline.py
   ```
3. **Run EDA**:
   ```bash
   python src/eda.py
   ```
4. **Train Model**:
   ```bash
   python src/model.py
   ```
5. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Evaluation Criteria Covered

- **Data Collection (1.5)**: Implemented realistic synthetic generation logic.
- **Data Cleaning & ETL (1.5)**: Handled missing values, duplicates, and feature engineering.
- **EDA & Visuals (2)**: Auto-generated 8 statistical plots.
- **ML Model & Evaluation (2)**: Regression model with R² score and feature importance.
- **Dashboard (1.5)**: Interactive Streamlit app with multiple pages.
- **Insights & Report (1.5)**: Documented actionable recommendations based on the data.
