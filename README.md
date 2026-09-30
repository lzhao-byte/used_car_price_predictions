# Intro to Predictive Analytics (Streamlit)

An interactive, multi-page Streamlit app that walks non-specialists through the full predictive-analytics lifecycle on a used-car price dataset. I built it as a hands-on lab for company-wide data-literacy training ("Data Days").

**Live demo:** https://intro-to-predictive-analytics.streamlit.app

## What you can do

| Page | What it covers |
|---|---|
| **Data Explorer** | Profile the raw data and look at distributions and relationships with interactive Plotly charts |
| **Data Prep** | Inspect the schema, handle nulls, trim outliers (IQR or percentile), remove duplicates, and standardize messy make/model names against a reference list using fuzzy matching (RapidFuzz) with a confidence score |
| **Feature Engineering** | Add, transform, and select features |
| **Model Training** | Train and compare linear models (Lasso / Ridge / ElasticNet), Random Forest, and XGBoost; view metrics (MAE, RMSE, R²) and predictions |
| **Monitoring (Simulation)** | Simulate production data over time to show how model performance degrades under data drift, and how you would monitor it |

## Tech

Python · Streamlit · Polars · scikit-learn · XGBoost · Plotly · RapidFuzz · Snowflake Snowpark (optional backend)

Data loads from local Parquet files (partitioned by price bin) by default. `utils/snowflake_functions.py` can read the same tables from Snowflake instead. Credentials come only from environment variables (`SNOWFLAKE_USER`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_AUTHENTICATOR`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run home.py
```

You can also open the repo in a GitHub Codespace, which uses the included `.devcontainer`. The app starts automatically.
