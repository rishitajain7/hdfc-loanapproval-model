import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "loan_applications_log.csv")

try:
    df = pd.read_csv(LOG_PATH)
except FileNotFoundError:
    print("⚠ loan_applications_log.csv not found — skipping")
    exit()

# Add loan_returned column if missing
if 'loan_returned' not in df.columns:
    df['loan_returned'] = ""

# Add trustworthiness_score column if missing
if 'trustworthiness_score' not in df.columns:
    df['trustworthiness_score'] = 50

df.to_csv(LOG_PATH, index=False)

print("✅ loan_applications_log.csv updated successfully")
print("Added columns: loan_returned, trustworthiness_score")
