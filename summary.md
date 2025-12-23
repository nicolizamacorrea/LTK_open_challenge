# Executive Summary: ShopFlow Returns Prediction & ROI Optimization

## Approach Overview
The goal of this project was to reduce ShopFlow's $400,000 monthly return costs by predicting return-prone orders and applying targeted $3 interventions. I transitioned from a simple baseline to an advanced **XGBoost** model incorporating hypothesis-driven feature engineering and robust monitoring.

## Key Findings
- **Feature Importance:** Absolute discount amount and the ratio of customer tenure to recency emerged as stronger predictors than raw demographic data.
- **Model Performance:** The improved XGBoost model increased the **ROC AUC to 0.61**, successfully identifying patterns that the baseline Logistic Regression missed.
- **ROI Sensitivity:** We found that even small improvements in model precision lead to significant savings when scaled to 100,000 monthly orders.

## Model Improvement & Validation
- **Feature Engineering:** Added `discount_amount`, `tenure_ratio`, and `is_high_value` flags.
- **Imbalance Handling:** Used `scale_pos_weight` to address the 25/75 class distribution.
- **Validation:** Confirmed minimal overfitting (Train ROC 0.73 vs Test ROC 0.61) and verified no data leakage.

## Business Impact
On a test set of 2,000 orders:
- **Baseline Savings:** $3,126
- **Improved Model Savings:** **$3,303**
- **Monthly Projection:** Scaled to 100,000 orders, this model is projected to save ShopFlow **~$165,150 per month**.

## Data Drift & Model Monitoring
To ensure long-term reliability:
- **Feature Drift:** Automated Kolmogorov-Smirnov (KS) tests to detect shifts in price or customer age distributions.
- **Concept Drift:** Weekly tracking of model ROC AUC; retraining triggers if performance drops below 0.58.
- **Prediction Drift:** Monitoring the distribution of predicted probabilities to ensure stable intervention volume.

## Deployment Recommendation
1. **Pilot Launch:** Deploy to 10% of users to validate the 35% reduction in return probability assumption.
2. **Threshold Selection:** Use a threshold of **0.38** to maximize total savings in the current environment.
3. **Automated Retraining:** Implement a monthly retraining pipeline using the most recent 12 months of transactional data.

**Final Model Package:** `final_model_package.pkl` (Model, Features, Threshold)
**Notebook:** `lizama_nicolas_challenge.ipynb`


## How to Run the Project

### 1. Environment Setup
It is recommended to use a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
Install the required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Running the Analysis
- **Jupyter Notebook:** Open `lizama_nicolas_challenge.ipynb` in VS Code or run `jupyter notebook` in your terminal. Ensure you have the datasets (`ecommerce_returns_train.csv` and `ecommerce_returns_test.csv`) in the same folder.
- **Python Script:** Run the automated improvement pipeline directly:
  ```bash
  python challenge_submission.py
  ```

### 4. Output Artifacts
- The script will generate `final_model_package.pkl`, which contains the trained model, feature names, and the optimal business threshold for deployment.
