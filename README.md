# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave, using the IBM Telco
Customer Churn dataset (7,043 customers, 21 columns). The project covers
cleaning, exploratory analysis, model comparison, imbalance handling,
threshold selection and SHAP interpretation.

## Key findings
- About 26.5% of customers churned. Churn is highest for month-to-month
  contracts, new customers, fiber optic users and electronic-check payers.
- Logistic regression performed about as well as tree-based models
  (cross-validated ROC-AUC about XX), so the simpler model was used.
- With class weights and a threshold chosen on out-of-fold predictions,
  recall for churners rose from 55.6% to 92.2% on the held-out test set,
  at a precision of 43.8%. The final model catches 345 of about 374 churners.
- The threshold favors recall (F2). It can be changed if retention offers
  are costly.

## Results (held-out test set)
| | Recall | Precision | F1 | Churners caught |
|---|---|---|---|---|
| Baseline logistic regression | 0.556 | 0.654 | 0.601 | 208 |
| Final model | 0.922 | 0.438 | 0.594 | 345 |

## Project structure
- `notebooks/01-09` : step-by-step analysis, in order
- `src/` : reusable preprocessing and model code
- `data/` : raw, cleaned and train/test CSV files
- `models/` : saved final model and threshold

## Method
1. Cleaned `TotalCharges` (11 blanks for new customers set to 0), dropped
   the ID column, merged redundant categories.
2. Stratified 80/20 train/test split; preprocessing fitted on training data
   only, inside scikit-learn pipelines to avoid leakage.
3. Compared logistic regression, decision tree, random forest and
   gradient boosting with 5-fold cross-validation.
4. Handled imbalance with class weights (SMOTE gave similar results).
5. Chose the decision threshold on out-of-fold predictions, then evaluated
   once on the test set.
6. Explained the model with SHAP.

## Run it
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    jupyter notebook

Dataset: Telco Customer Churn (IBM sample data) on Kaggle.