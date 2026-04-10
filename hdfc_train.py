# ==========================================
# HDFC Loan Approval - Training Script
# ==========================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("loan_approval_dataset.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print("-----------------------------------")

# -----------------------------
# 2. Data Preparation
# -----------------------------

# Remove irrelevant columns if they exist
df = df.drop(columns=["loan_id", "name", "pan_card"], errors="ignore")

# Keep only numeric columns (removes strings automatically)
df_numeric = df.select_dtypes(include=["int64", "float64"])

# Target variable
y = df_numeric["loan_status_enc"]

# Feature variables
X = df_numeric.drop(columns=["loan_status_enc"])

print("Features Selected:", list(X.columns))
print("-----------------------------------")

# -----------------------------
# 3. Train-Test Split (80/20)
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Model Training
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")
print("-----------------------------------")

# -----------------------------
# 5. Model Evaluation
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Save Model & Test Data
# -----------------------------

joblib.dump(model, "loan_model.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("\nModel saved as loan_model.pkl")
print("Test data saved successfully")