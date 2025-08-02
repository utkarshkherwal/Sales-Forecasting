# Sales Forecaster

## Overview

This project predicts weekly sales for retail stores using historical sales data, store information, and external features such as holidays and promotions. By leveraging advanced data processing and machine learning models (Random Forest, XGBoost), the project delivers accurate sales forecasts to help retailers optimize inventory, staffing, and marketing decisions.

## Data Source

The dataset used for this project comes from the [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) competition hosted on Kaggle. It includes detailed historical sales records across multiple departments and stores, along with features such as markdown events, holiday flags, CPI, temperature, and fuel prices. This rich dataset allows for modeling real-world retail dynamics and improving the accuracy of store-level sales forecasting.

## Features

- **Data Integration:** Merges sales, store, and promotional data for holistic analysis.
- **Feature Engineering:** Extracts time-based features, encodes categories, and highlights holidays/markdown events.
- **Trend Analysis:** Computes lag features, moving averages, and major holiday impact.
- **Quota & Deviation:** Calculates sales quotas and deviations for performance tracking.
- **Machine Learning:** Implements Random Forest and XGBoost regressors for robust forecasting.
- **Visualizations:** Offers insights into feature importance, error distribution, and prediction comparisons.

## Real-World Impact

Accurate store-level sales forecasting enables:
- Better inventory and supply chain management
- Reduced stockouts and overstock
- Improved promotional planning
- Data-driven business decisions
