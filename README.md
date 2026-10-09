# -credit-risk-xgboost
Interpretable credit risk modeling with XGBoost and SHAP
"""
Credit Risk Modeling with XGBoost
Interpretable ML project for credit default prediction.
Author: Alireza Vafaeian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target"
]

df = pd.read_csv(url, sep=" ", names=columns)
print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Target: 1 = good, 2 = bad. Convert to 0/1 (1 = default).
df["target"] = df["target"].map({1: 0, 2: 1})
print(f"Default rate: {df['target'].mean():.2%}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical features: {len(categorical_cols)}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"After encoding — shape: {df_encoded.shape}")

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# 3. MODEL TRAINING
# ============================================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
print("\n" + "=" * 60)
print("MODEL TRAINED")
print("=" * 60)

# ============================================================
# 4. EVALUATION
# ============================================================
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Good", "Default"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                Predicted Good  Predicted Default")
print(f"Actual Good          {cm[0,0]:4d}             {cm[0,1]:4d}")
print(f"Actual Default       {cm[1,0]:4d}             {cm[1,1]:4d}")

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title("ROC Curve — Credit Default Prediction")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ============================================================
# 5. INTERPRETABILITY WITH SHAP
# ============================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()

# Feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()

# Top 10 features
mean_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_shap)[-10:][::-1]
print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (by mean |SHAP|)")
print("=" * 60)
for i, idx in enumerate(top_idx, 1):
    print(f"{i:2d}. {X_test.columns[idx]:45s} {mean_shap[idx]:.4f}")

print("\n" + "=" * 60)
print("Project completed successfully.")
print("Outputs: roc_curve.png, shap_summary.png, shap_importance.png")
print("=" * 60)