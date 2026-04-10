import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

df = pd.read_csv(DATASET_PATH)

# Add loan_returned column if missing
if 'loan_returned' not in df.columns:
    df['loan_returned'] = ""

# Add trustworthiness_score column if missing
if 'trustworthiness_score' not in df.columns:
    # Default: 50 for first-time applicants
    df['trustworthiness_score'] = 50

df.to_csv(DATASET_PATH, index=False)

print("✅ loan_approval_dataset.csv updated successfully")
print("Added columns: loan_returned, trustworthiness_score")
