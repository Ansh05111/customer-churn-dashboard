"""
Customer Churn Prediction — Model Training Pipeline
------------------------------------------------------
Loads the Telco Customer Churn dataset, cleans and encodes it,
trains and compares three classifiers (Logistic Regression, Random
Forest, XGBoost), tunes XGBoost via GridSearchCV optimizing for F1,
and saves the final model plus SHAP explainability artifacts for the
Streamlit dashboard to use.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------- Step 3-5: Load, clean, and encode data ----------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop(columns=["customerID"])

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

print(df[binary_cols].head())

categorical_cols = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod"
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(df.shape)
print(df.columns.tolist())


# ---------------- Step 6: Train/test split ----------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train churn rate:", y_train.mean())
print("Test churn rate:", y_test.mean())


# ---------------- Step 7-8: Logistic Regression (scaled baseline) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# ---------------- Step 9-10: Random Forest ----------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)  # unscaled — tree models don't need scaling

y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))


# ---------------- Step 11: XGBoost (untuned baseline) ----------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))


# ---------------- Step 12: Confusion matrix (untuned XGBoost) ----------------
cm = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix (XGBoost):")
print(cm)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.savefig("confusion_matrix_xgb.png")
plt.close()
print("Saved confusion_matrix_xgb.png")


# ---------------- Step 13-14: Hyperparameter tuning (optimized for F1) ----------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

xgb_base = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

grid_search_f1 = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search_f1.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Results (optimized for F1) ---")
print("Best parameters:", grid_search_f1.best_params_)
print("Best cross-validated F1:", grid_search_f1.best_score_)

best_xgb_f1 = grid_search_f1.best_estimator_
y_pred_best_f1 = best_xgb_f1.predict(X_test)

print("\n--- Tuned XGBoost F1-optimized (Test Set) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best_f1))
print("Precision:", precision_score(y_test, y_pred_best_f1))
print("Recall:", recall_score(y_test, y_pred_best_f1))
print("F1 Score:", f1_score(y_test, y_pred_best_f1))


# ---------------- Step 15: Save final model ----------------
joblib.dump(best_xgb_f1, "final_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nFinal model saved as final_model.pkl")


# ---------------- Confusion matrix (final tuned model) ----------------
cm_final = confusion_matrix(y_test, y_pred_best_f1)
print("\nConfusion Matrix (Final F1-tuned XGBoost):")
print(cm_final)

plt.figure()
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=["No Churn", "Churn"])
disp_final.plot(cmap="Blues")
plt.title("Final Model Confusion Matrix")
plt.savefig("confusion_matrix_final.png")
plt.close()
print("Saved confusion_matrix_final.png")


# ---------------- Step 16-18: SHAP explainability ----------------
explainer = shap.TreeExplainer(best_xgb_f1)
shap_values = explainer.shap_values(X_test)

print("\nSHAP values calculated")
print("Shape of shap_values:", shap_values.shape)
print("Shape of X_test:", X_test.shape)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_global_importance.png")
plt.close()
print("Saved shap_global_importance.png")

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("Saved shap_summary.png")


# ---------------- Step 19: Individual customer explanation (example) ----------------
sample_idx = 0
sample_customer = X_test.iloc[[sample_idx]]
sample_prediction = best_xgb_f1.predict(sample_customer)[0]
sample_probability = best_xgb_f1.predict_proba(sample_customer)[0][1]

print(f"\nSample customer prediction: {'Churn' if sample_prediction == 1 else 'No Churn'}")
print(f"Churn probability: {sample_probability:.2%}")

plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    sample_customer,
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("shap_individual_example.png")
plt.close()
print("Saved shap_individual_example.png")