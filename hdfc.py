import os
import sys
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# PATH SETUP
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
LOG_PATH = os.path.join(BASE_DIR, "loan_applications_log.csv")

THRESHOLD = 0.40

# ==================================================
# LOAD DATASET
# ==================================================
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower()

# Ensure new columns exist
for col, default in {
    "loan_returned": "",
    "trustworthiness_score": 50
}.items():
    if col not in df.columns:
        df[col] = default

# Clean categorical fields
for col in ['education', 'self_employed', 'loan_status']:
    df[col] = df[col].astype(str).str.strip()

# Encode for ML
df['education_enc'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed_enc'] = df['self_employed'].map({'Yes': 1, 'No': 0})
df['loan_status_enc'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

FEATURE_COLS = [
    'no_of_dependents',
    'education_enc',
    'self_employed_enc',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

X = df[FEATURE_COLS]
y = df['loan_status_enc']

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def estimate_loan_return(row):
    monthly_emi = row['loan_amount'] / row['loan_term']
    monthly_capacity = (row['income_annum'] / 12) * 0.4

    total_assets = (
        row['residential_assets_value']
        + row['commercial_assets_value']
        + row['luxury_assets_value']
        + row['bank_asset_value']
    )

    asset_backup = total_assets >= row['loan_amount'] * 1.2

    return "Yes" if monthly_capacity >= monthly_emi and asset_backup else "No"


def update_trust_score(current_score, loan_returned):
    if loan_returned == "Yes":
        return min(100, current_score + 15)
    else:
        return max(0, current_score - 25)


def predict_loan(row):
    features = row[FEATURE_COLS].to_frame().T
    features = pd.DataFrame(imputer.transform(features), columns=FEATURE_COLS)
    prob = model.predict_proba(features)[0][1]
    return ("Approved" if prob >= THRESHOLD else "Rejected"), round(prob * 100, 2)

# ==================================================
# NEW APPLICATION
# ==================================================
def new_application():
    global df

    print("\n--- NEW LOAN APPLICATION ---")

    name = input("Applicant Name: ").strip()
    pan = input("PAN Card Number: ").strip().upper()

    previous = df[df['pan_card'].str.upper() == pan]

    # PAN exists → validate name
    if not previous.empty:
        stored_name = previous.iloc[0]['name'].strip().lower()
        if name.lower() != stored_name:
            print("\nPAN holder name mismatch. Entry cancelled.\n")
            return

    # ============================
    # DEFAULT VALUES (from last record if exists)
    # ============================
    if not previous.empty:
        last = previous.iloc[-1]

        print("\nExisting PAN detected.")

        same_assets_income = int(
            input("Are income and assets same as previous record? (1=Yes, 0=No): ")
        )

        if same_assets_income == 1:
            income_annum = last['income_annum']
            residential_assets_value = last['residential_assets_value']
            commercial_assets_value = last['commercial_assets_value']
            luxury_assets_value = last['luxury_assets_value']
            bank_asset_value = last['bank_asset_value']
        else:
            income_annum = float(input("Annual income: "))
            residential_assets_value = float(input("Residential assets: "))
            commercial_assets_value = float(input("Commercial assets: "))
            luxury_assets_value = float(input("Luxury assets: "))
            bank_asset_value = float(input("Bank assets: "))

        same_dependents = int(
            input("Are number of dependents same as previous record? (1=Yes, 0=No): ")
        )

        if same_dependents == 1:
            no_of_dependents = last['no_of_dependents']
        else:
            no_of_dependents = int(input("Number of dependents: "))

        # Education & employment assumed unchanged
        education = last['education']
        self_employed = last['self_employed']
        cibil_score = last['cibil_score']
        trust_score = last['trustworthiness_score']

    else:
        # ============================
        # NEW PAN — FULL INPUT
        # ============================
        no_of_dependents = int(input("Number of dependents: "))
        education = input("Education (Graduate/Not Graduate): ").strip()
        self_employed = input("Self Employed (Yes/No): ").strip()
        income_annum = float(input("Annual income: "))
        cibil_score = int(input("CIBIL score: "))
        residential_assets_value = float(input("Residential assets: "))
        commercial_assets_value = float(input("Commercial assets: "))
        luxury_assets_value = float(input("Luxury assets: "))
        bank_asset_value = float(input("Bank assets: "))
        trust_score = 50  # Initial trust score

    # ============================
    # LOAN-SPECIFIC INPUT (ALWAYS ASK)
    # ============================
    loan_amount = float(input("Loan amount: "))
    loan_term = int(input("Loan term (months): "))

    # ============================
    # CREATE RECORD
    # ============================
    record = {
        'loan_id': df['loan_id'].max() + 1 if not df.empty else 1,
        'name': name,
        'pan_card': pan,
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'loan_returned': "",
        'trustworthiness_score': trust_score
    }

    record['education_enc'] = 1 if education == "Graduate" else 0
    record['self_employed_enc'] = 1 if self_employed == "Yes" else 0

    temp_row = pd.Series(record)

    # ============================
    # UPDATE PREVIOUS LOAN (if repeat PAN)
    # ============================
    if not previous.empty:
        prev_idx = previous.index[-1]
        returned = estimate_loan_return(df.loc[prev_idx])
        df.at[prev_idx, 'loan_returned'] = returned

        new_score = update_trust_score(
            df.loc[prev_idx, 'trustworthiness_score'],
            returned
        )
        df.at[prev_idx, 'trustworthiness_score'] = new_score
        record['trustworthiness_score'] = new_score
    

    decision, prob = predict_loan(temp_row)

# Override ML decision if trustworthiness too low
    if record['trustworthiness_score'] < 15:
        decision = "Rejected"
        prob = 0.0

    record['loan_status'] = decision
    record['loan_status_enc'] = 1 if decision == "Approved" else 0

    df.loc[len(df)] = record
    df.to_csv(DATASET_PATH, index=False)

    print("\nApplication saved successfully")
    print(f"Loan Status: {decision}")
    print(f"Approval Probability: {prob}%")
    print(f"Trustworthiness Score: {record['trustworthiness_score']}")

    return

    print("\n--- NEW LOAN APPLICATION ---")

    name = input("Applicant Name: ").strip()
    pan = input("PAN Card Number: ").strip().upper()

    previous = df[df['pan_card'].str.upper() == pan]

    # PAN exists → validate name
    if not previous.empty:
        stored_name = previous.iloc[0]['name'].strip().lower()
        if name.lower() != stored_name:
            print("\nPAN holder name mismatch. Entry cancelled.\n")
            return

    # Input details
    record = {
        'loan_id': df['loan_id'].max() + 1 if not df.empty else 1,
        'name': name,
        'pan_card': pan,
        'no_of_dependents': int(input("Dependents: ")),
        'education': input("Education (Graduate/Not Graduate): ").strip(),
        'self_employed': input("Self Employed (Yes/No): ").strip(),
        'income_annum': float(input("Annual income: ")),
        'loan_amount': float(input("Loan amount: ")),
        'loan_term': int(input("Loan term (months): ")),
        'cibil_score': int(input("CIBIL score: ")),
        'residential_assets_value': float(input("Residential assets: ")),
        'commercial_assets_value': float(input("Commercial assets: ")),
        'luxury_assets_value': float(input("Luxury assets: ")),
        'bank_asset_value': float(input("Bank assets: ")),
        'loan_returned': "",
        'trustworthiness_score': previous.iloc[-1]['trustworthiness_score'] if not previous.empty else 50
    }

    record['education_enc'] = 1 if record['education'] == "Graduate" else 0
    record['self_employed_enc'] = 1 if record['self_employed'] == "Yes" else 0

    temp_row = pd.Series(record)

    # If repeat PAN → evaluate previous loan repayment
    if not previous.empty:
        prev_idx = previous.index[-1]
        returned = estimate_loan_return(df.loc[prev_idx])
        df.at[prev_idx, 'loan_returned'] = returned

        new_score = update_trust_score(
            df.loc[prev_idx, 'trustworthiness_score'],
            returned
        )
        df.at[prev_idx, 'trustworthiness_score'] = new_score
        record['trustworthiness_score'] = new_score

    decision, prob = predict_loan(temp_row)
    record['loan_status'] = decision
    record['loan_status_enc'] = 1 if decision == "Approved" else 0

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False)

    # Log file update
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'name': name,
        'pan_card': pan,
        'loan_status': decision,
        'approval_probability': prob,
        'loan_returned': record['loan_returned'],
        'trustworthiness_score': record['trustworthiness_score']
    }

    try:
        log = pd.read_csv(LOG_PATH)
        log = pd.concat([log, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        log = pd.DataFrame([log_entry])

    log.to_csv(LOG_PATH, index=False)

    print("\nApplication saved successfully")
    print(f"Loan Status: {decision}")
    print(f"Approval Probability: {prob}%")
    print(f"Trustworthiness Score: {record['trustworthiness_score']}")

# ==================================================
# SEARCH
# ==================================================
def search_application():
    pan = input("\nEnter PAN to search: ").strip().upper()
    records = df[df['pan_card'].str.upper() == pan]

    if records.empty:
        print("No record found\n")
        return

    print("\n====== APPLICANT RECORD(S) ======")

    for _, record in records.iterrows():
        print("\n-------------------------------")
        print(f"Loan ID: {record['loan_id']}")
        print(f"Applicant Name: {record['name']}")
        print(f"PAN Card Number: {record['pan_card']}")
        print(f"Dependents: {record['no_of_dependents']}")
        print(f"Education: {record['education']}")
        print(f"Self Employed: {record['self_employed']}")
        print(f"Annual Income: {record['income_annum']}")
        print(f"Loan Amount: {record['loan_amount']}")
        print(f"Loan Term (months): {record['loan_term']}")
        print(f"CIBIL Score: {record['cibil_score']} (Read-only)")
        print(f"Residential Assets: {record['residential_assets_value']}")
        print(f"Commercial Assets: {record['commercial_assets_value']}")
        print(f"Luxury Assets: {record['luxury_assets_value']}")
        print(f"Bank Assets: {record['bank_asset_value']}")
        print(f"Loan Status: {record['loan_status']}")
        print(f"Loan Returned: {record['loan_returned']}")
        print(f"Trustworthiness Score: {record['trustworthiness_score']}")
        print("-------------------------------")

    print("\n====== END OF RECORD ======\n")

# ==================================================
# MAIN MENU
# ==================================================
while True:
    print("\n====== HDFC LOAN SYSTEM ======")
    print("1. Search application")
    print("2. New loan application")
    print("3. Exit")

    choice = input("Choose option (1/2/3): ").strip()

    if choice == '1':
        search_application()
    elif choice == '2':
        new_application()
    elif choice == '3':
        print("Exiting system")
        sys.exit()
    else:
        print("Invalid choice\n")
