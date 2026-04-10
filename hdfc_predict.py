# ==========================================
# HDFC Loan Approval - Prediction Script
# ==========================================

import joblib
import pandas as pd

# -----------------------------
# 1. Load Saved Files
# -----------------------------

model = joblib.load("loan_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

print("Model Loaded Successfully")
print("-----------------------------------")

# -----------------------------
# 2. Feature Importance
# -----------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X_test.columns
).sort_values(ascending=False)

print("Top 5 Most Important Features:")
print(importance.head(5))
print("-----------------------------------")

# -----------------------------
# 3. Prediction Display Function
# -----------------------------

def show_prediction(index):

    applicant = X_test.iloc[index]
    actual = y_test.iloc[index]

    prediction = model.predict([applicant])[0]
    probability = model.predict_proba([applicant])[0]

    approval_probability = probability[1]
    confidence = max(probability)

    print("\n=================================")
    print(f"APPLICANT #{index}")
    print("---------------------------------")
    print("Model Prediction:",
          "Approved" if prediction == 1 else "Rejected")
    print("Approval Probability:",
          round(approval_probability * 100, 2), "%")
    print("Model Confidence:",
          round(confidence * 100, 2), "%")
    print("Actual Status:",
          "Approved" if actual == 1 else "Rejected")

    if prediction == actual:
        print("Prediction Accuracy: CORRECT")
    else:
        print("Prediction Accuracy: WRONG")

    print("=================================")

# -----------------------------
# 4. Show Sample Predictions
# -----------------------------

show_prediction(0)
show_prediction(1)
show_prediction(2)