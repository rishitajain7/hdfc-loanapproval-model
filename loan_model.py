# # import pandas as pd
# # import numpy as np

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # import joblib


# # # 1. Load dataset
# # df = pd.read_csv("loan_data_3500.csv")

# # # Clean column names
# # df.columns = df.columns.str.strip().str.lower()

# # print("Dataset Shape:", df.shape)
# # print(df.head())


# # # 2. Encode categorical columns
# # le = LabelEncoder()
# # df['education'] = le.fit_transform(df['education'])
# # df['self_employed'] = le.fit_transform(df['self_employed'])
# # df['loan_status'] = le.fit_transform(df['loan_status'])  # Approved=1, Rejected=0


# # # 3. Split features & target
# # X = df.drop(['loan_status', 'loan_id'], axis=1)
# # y = df['loan_status']


# # # 4. Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42
# # )


# # # 5. Feature scaling (for Logistic Regression)
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)


# # # 6. Logistic Regression
# # lr = LogisticRegression(max_iter=1000)
# # lr.fit(X_train_scaled, y_train)
# # lr_pred = lr.predict(X_test_scaled)

# # print("\n--- Logistic Regression ---")
# # print("Accuracy:", accuracy_score(y_test, lr_pred))


# # # 7. Random Forest (BEST MODEL)
# # rf = RandomForestClassifier(n_estimators=200, random_state=42)
# # rf.fit(X_train, y_train)
# # rf_pred = rf.predict(X_test)

# # print("\n--- Random Forest ---")
# # print("Accuracy:", accuracy_score(y_test, rf_pred))
# # print(confusion_matrix(y_test, rf_pred))
# # print(classification_report(y_test, rf_pred))


# # # 8. Save model
# # joblib.dump(rf, "loan_approval_model.pkl")
# # joblib.dump(scaler, "scaler.pkl")

# # print("\nModel & scaler saved successfully")


# # # 9. Predict for a new applicant
# # # new_applicant = pd.DataFrame([{
# # #     'no_of_dependents': 2,
# # #     'education': 1,              # Graduate
# # #     'self_employed': 0,           # No
# # #     'income_annum': 800000,
# # #     'loan_amount': 200000,
# # #     'loan_term': 360,
# # #     'cibil_score': 750,
# # #     'residential_assets_value': 3000000,
# # #     'commercial_assets_value': 0,
# # #     'luxury_assets_value': 0,
# # #     'bank_asset_value': 500000
# # # }])

# # print("\n--- Enter Applicant Details ---")

# # no_of_dependents = int(input("Number of dependents: "))
# # education = int(input("Education (1 = Graduate, 0 = Not Graduate): "))
# # self_employed = int(input("Self employed? (1 = Yes, 0 = No): "))
# # income_annum = float(input("Annual income: "))
# # loan_amount = float(input("Loan amount: "))
# # loan_term = int(input("Loan term (in months): "))
# # cibil_score = int(input("CIBIL score: "))
# # residential_assets_value = float(input("Residential assets value: "))
# # commercial_assets_value = float(input("Commercial assets value: "))
# # luxury_assets_value = float(input("Luxury assets value: "))
# # bank_asset_value = float(input("Bank asset value: "))

# # new_applicant = pd.DataFrame([{
# #     'no_of_dependents': no_of_dependents,
# #     'education': education,
# #     'self_employed': self_employed,
# #     'income_annum': income_annum,
# #     'loan_amount': loan_amount,
# #     'loan_term': loan_term,
# #     'cibil_score': cibil_score,
# #     'residential_assets_value': residential_assets_value,
# #     'commercial_assets_value': commercial_assets_value,
# #     'luxury_assets_value': luxury_assets_value,
# #     'bank_asset_value': bank_asset_value
# # }])

# # prediction = rf.predict(new_applicant)

# # print("\n--- Loan Prediction Result ---")
# # if prediction[0] == 1:
# #     print("Loan Approved")
# # else:
# #     print("Loan Rejected")


# # prediction = rf.predict(new_applicant)

# # print("\n--- New Applicant Result ---")
# # print("Loan Approved " if prediction[0] == 1 else "Loan Rejected ")


# # import pandas as pd
# # import numpy as np

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # import joblib

# # # =========================
# # # 1. Load Dataset
# # # =========================
# # df = pd.read_csv("loan_data_3500.csv")

# # # Clean column names
# # df.columns = df.columns.str.strip().str.lower()

# # print("Dataset Shape:", df.shape)
# # print(df.head())

# # # =========================
# # # 2. Manual Encoding (NO LabelEncoder ❌)
# # # =========================
# # # This avoids ALL encoding mismatch bugs

# # df['education'] = df['education'].map({
# #     'Graduate': 1,
# #     'Not Graduate': 0
# # })

# # df['self_employed'] = df['self_employed'].map({
# #     'Yes': 1,
# #     'No': 0
# # })

# # df['loan_status'] = df['loan_status'].map({
# #     'Approved': 1,
# #     'Rejected': 0
# # })

# # # =========================
# # # 3. Features & Target
# # # =========================
# # X = df.drop(['loan_status', 'loan_id'], axis=1)
# # y = df['loan_status']

# # print("\nFeature Columns:")
# # print(X.columns)

# # # =========================
# # # 4. Train-Test Split
# # # =========================
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42
# # )

# # # =========================
# # # 5. Scaling (ONLY for Logistic Regression)
# # # =========================
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # =========================
# # # 6. Logistic Regression
# # # =========================
# # lr = LogisticRegression(max_iter=1000)
# # lr.fit(X_train_scaled, y_train)
# # lr_pred = lr.predict(X_test_scaled)

# # print("\n--- Logistic Regression ---")
# # print("Accuracy:", accuracy_score(y_test, lr_pred))

# # # =========================
# # # 7. Random Forest (MAIN MODEL)
# # # =========================
# # rf = RandomForestClassifier(
# #     n_estimators=200,
# #     random_state=42
# # )

# # rf.fit(X_train, y_train)
# # rf_pred = rf.predict(X_test)

# # print("\n--- Random Forest ---")
# # print("Accuracy:", accuracy_score(y_test, rf_pred))
# # print(confusion_matrix(y_test, rf_pred))
# # print(classification_report(y_test, rf_pred))

# # # =========================
# # # 8. Feature Importance (VIVA GOLD)
# # # =========================
# # importance = pd.DataFrame({
# #     "Feature": X.columns,
# #     "Importance": rf.feature_importances_
# # }).sort_values(by="Importance", ascending=False)

# # print("\n--- Feature Importance ---")
# # print(importance)

# # # =========================
# # # 9. Save Model & Scaler
# # # =========================
# # joblib.dump(rf, "loan_approval_model.pkl")
# # joblib.dump(scaler, "scaler.pkl")

# # print("\nModel & scaler saved successfully")

# # # =========================
# # # 10. USER INPUT FOR NEW APPLICANT
# # # =========================
# # print("\n--- Enter Applicant Details ---")

# # no_of_dependents = int(input("Number of dependents: "))
# # education = int(input("Education (1 = Graduate, 0 = Not Graduate): "))
# # self_employed = int(input("Self employed? (1 = Yes, 0 = No): "))
# # income_annum = float(input("Annual income: "))
# # loan_amount = float(input("Loan amount: "))
# # loan_term = int(input("Loan term: "))
# # cibil_score = int(input("CIBIL score: "))
# # residential_assets_value = float(input("Residential assets value: "))
# # commercial_assets_value = float(input("Commercial assets value: "))
# # luxury_assets_value = float(input("Luxury assets value: "))
# # bank_asset_value = float(input("Bank asset value: "))

# # # Create DataFrame in EXACT SAME ORDER as training
# # new_applicant = pd.DataFrame([[
# #     no_of_dependents,
# #     education,
# #     self_employed,
# #     income_annum,
# #     loan_amount,
# #     loan_term,
# #     cibil_score,
# #     residential_assets_value,
# #     commercial_assets_value,
# #     luxury_assets_value,
# #     bank_asset_value
# # ]], columns=X.columns)

# # print("\nApplicant Data:")
# # print(new_applicant)

# # # =========================
# # # 11. Prediction
# # # =========================
# # prediction = rf.predict(new_applicant)
# # probability = rf.predict_proba(new_applicant)

# # print("\n--- Loan Prediction Result ---")
# # if prediction[0] == 1:
# #     print("✅ Loan Approved")
# # else:
# #     print("❌ Loan Rejected")

# # print(f"Approval Probability: {probability[0][1]*100:.2f}%")

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# import joblib

# # =========================
# # 1. Load Dataset
# # =========================
# df = pd.read_csv("loan_data_3500.csv")

# df.columns = df.columns.str.strip().str.lower()

# print("Dataset Shape:", df.shape)
# print(df.head())

# # =========================
# # 2. Manual Encoding (NO LabelEncoder)
# # =========================
# df['education'] = df['education'].map({
#     'Graduate': 1,
#     'Not Graduate': 0
# })

# df['self_employed'] = df['self_employed'].map({
#     'Yes': 1,
#     'No': 0
# })

# df['loan_status'] = df['loan_status'].map({
#     'Approved': 1,
#     'Rejected': 0
# })

# # =========================
# # 3. Features & Target
# # =========================
# X = df.drop(['loan_status', 'loan_id'], axis=1)
# y = df['loan_status']

# print("\nFeature Columns:")
# print(X.columns.tolist())

# # =========================
# # 4. Train-Test Split
# # =========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # =========================
# # 5. Scaling (Logistic Regression only)
# # =========================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # =========================
# # 6. Logistic Regression
# # =========================
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train_scaled, y_train)
# lr_pred = lr.predict(X_test_scaled)

# print("\n--- Logistic Regression ---")
# print("Accuracy:", accuracy_score(y_test, lr_pred))

# # =========================
# # 7. Random Forest (MAIN MODEL)
# # =========================
# rf = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42
# )

# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# print("\n--- Random Forest ---")
# print("Accuracy:", accuracy_score(y_test, rf_pred))
# print(confusion_matrix(y_test, rf_pred))
# print(classification_report(y_test, rf_pred))

# # =========================
# # 8. Feature Importance (Viva Gold)
# # =========================
# importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": rf.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# print("\n--- Feature Importance ---")
# print(importance)

# # =========================
# # 9. Save Model
# # =========================
# joblib.dump(rf, "loan_approval_model.pkl")
# joblib.dump(scaler, "scaler.pkl")

# print("\nModel & scaler saved successfully")

# # =========================
# # 10. USER INPUT
# # =========================
# print("\n--- Enter Applicant Details ---")

# no_of_dependents = int(input("Number of dependents: "))
# education = int(input("Education (1 = Graduate, 0 = Not Graduate): "))
# self_employed = int(input("Self employed? (1 = Yes, 0 = No): "))
# income_annum = float(input("Annual income: "))
# loan_amount = float(input("Loan amount: "))
# loan_term = int(input("Loan term (months): "))
# cibil_score = int(input("CIBIL score: "))
# residential_assets_value = float(input("Residential assets value: "))
# commercial_assets_value = float(input("Commercial assets value: "))
# luxury_assets_value = float(input("Luxury assets value: "))
# bank_asset_value = float(input("Bank asset value: "))

# # Ensure SAME feature order
# new_applicant = pd.DataFrame([[
#     no_of_dependents,
#     education,
#     self_employed,
#     income_annum,
#     loan_amount,
#     loan_term,
#     cibil_score,
#     residential_assets_value,
#     commercial_assets_value,
#     luxury_assets_value,
#     bank_asset_value
# ]], columns=X.columns)

# print("\nApplicant Data:")
# print(new_applicant)

# # =========================
# # 11. PROBABILITY-BASED DECISION (FIX)
# # =========================
# approval_prob = rf.predict_proba(new_applicant)[0][1]

# THRESHOLD = 0.40   # Tunable risk threshold

# print("\n--- Loan Prediction Result ---")
# print(f"Approval Probability: {approval_prob*100:.2f}%")

# if approval_prob >= THRESHOLD:
#     print("✅ Loan Approved")
# else:
#     print("❌ Loan Rejected")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("loan_data_3500.csv")

df.columns = df.columns.str.strip().str.lower()

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# 2. Clean categorical text
# =========================
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

# =========================
# 3. Manual Encoding (SAFE)
# =========================
df['education'] = df['education'].map({
    'Graduate': 1,
    'Not Graduate': 0
})

df['self_employed'] = df['self_employed'].map({
    'Yes': 1,
    'No': 0
})

df['loan_status'] = df['loan_status'].map({
    'Approved': 1,
    'Rejected': 0
})

# =========================
# 4. Separate Features & Target
# =========================
X = df.drop(['loan_status', 'loan_id'], axis=1)
y = df['loan_status']

print("\nMissing values before imputation:")
print(X.isnull().sum())

# =========================
# 5. HANDLE MISSING VALUES (CRITICAL FIX)
# =========================
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("\nMissing values after imputation:")
print(X.isnull().sum())

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. Scaling (for Logistic Regression)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 8. Logistic Regression
# =========================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))

# =========================
# 9. Random Forest (MAIN MODEL)
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# =========================
# 10. Feature Importance
# =========================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance)

# =========================
# 11. Save Everything
# =========================
joblib.dump(rf, "loan_approval_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\nModel, scaler & imputer saved successfully")

# =========================
# 12. USER INPUT
# =========================
print("\n--- Enter Applicant Details ---")

no_of_dependents = int(input("Number of dependents: "))
education = int(input("Education (1 = Graduate, 0 = Not Graduate): "))
self_employed = int(input("Self employed? (1 = Yes, 0 = No): "))
income_annum = float(input("Annual income: "))
loan_amount = float(input("Loan amount: "))
loan_term = int(input("Loan term: "))
cibil_score = int(input("CIBIL score: "))
residential_assets_value = float(input("Residential assets value: "))
commercial_assets_value = float(input("Commercial assets value: "))
luxury_assets_value = float(input("Luxury assets value: "))
bank_asset_value = float(input("Bank asset value: "))

new_applicant = pd.DataFrame([[
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=X.columns)

# Apply SAME imputer
new_applicant = pd.DataFrame(
    imputer.transform(new_applicant),
    columns=X.columns
)

# =========================
# 13. PROBABILITY-BASED DECISION
# =========================
approval_prob = rf.predict_proba(new_applicant)[0][1]
THRESHOLD = 0.40

print("\n--- Loan Prediction Result ---")
print(f"Approval Probability: {approval_prob*100:.2f}%")

if approval_prob >= THRESHOLD:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")
