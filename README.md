# 📊 Customer Churn Prediction Dashboard

An interactive dashboard that predicts customer churn using XGBoost, with SHAP-based
explainability for both global feature importance and individual predictions.

**Live demo:**  https://churn-prediction-xgboost.streamlit.app/

## Key Findings

- **Tenure is the strongest churn predictor** — customers in their first few months
  are far more likely to leave than long-tenured customers.
- **Contract length matters significantly** — month-to-month customers churn at a
  much higher rate than those on one- or two-year contracts.
- **Fiber optic internet customers churn more** than DSL customers — a
  counterintuitive finding, possibly related to price sensitivity or service
  quality concerns in that segment.
- **Electronic check users churn more** than customers using automatic payment
  methods.
- The final model catches **80% of actual churners** (Recall), a meaningful
  improvement over an untuned baseline that caught only 70%.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 80.7% | 65.8% | 56.7% | 60.9% |
| Random Forest (balanced) | 76.9% | 55.5% | 64.4% | 59.7% |
| XGBoost (untuned) | 75.7% | 53.3% | 69.5% | 60.3% |
| **XGBoost (F1-tuned, final)** | **74.7%** | **51.6%** | **80.0%** | **62.7%** |

The final model was chosen via 5-fold cross-validated GridSearchCV, optimizing for
F1 score rather than accuracy, since this dataset has significant class imbalance
(~73% No Churn / 27% Churn) — a naive always-predict-No model would already score
~73% accuracy while catching zero real churners.

## Known Limitations

- The live single-customer prediction form collects the most influential features
  (per SHAP analysis) but uses sensible defaults for a few less-impactful original
  features (e.g., streaming service subscriptions) rather than collecting all 19.
- Headline metrics are reported from a single stratified train/test split; 5-fold
  CV was used internally during hyperparameter tuning but not re-aggregated for
  the final reported numbers.
- Batch upload validates required columns exist, but does not currently validate
  that categorical values match the training data's known categories.

## Tech Stack

| Layer | Tool |
|---|---|
| Modeling | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Charts | Plotly, Matplotlib |
| Deployment | Streamlit Community Cloud |

## Project Structure