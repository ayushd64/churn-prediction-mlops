"""
Simulate a drifted batch of 'new' customers for the monitoring demo.

Takes the test set and shifts several distributions (shorter tenure, higher
charges, more month-to-month/fiber/electronic-check) to mimic a new customer
segment the model wasn't trained on.
"""

from pathlib import Path

import pandas as pd

src = Path("data/processed/test.csv")
dst = Path("data/processed/drifted_batch.csv")

df = pd.read_csv(src)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

df["tenure"] = (df["tenure"] * 0.25).astype(int)
df["MonthlyCharges"] = df["MonthlyCharges"] + 45
df["TotalCharges"] = df["TotalCharges"] * 0.3
df["Contract"] = "Month-to-month"
df["PaymentMethod"] = "Electronic check"
df["InternetService"] = "Fiber optic"

df.to_csv(dst, index=False)
print(f"Drifted batch written to {dst} ({len(df)} rows)")
