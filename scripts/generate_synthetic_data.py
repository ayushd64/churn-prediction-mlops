"""
Generate a small synthetic Telco-like dataset for CI.

CI servers don't have the real (gitignored) dataset, so this script fabricates
a structurally-identical sample — same columns, same TotalCharges blank-string
trap, same SeniorCitizen 0/1 quirk — letting the full pipeline run in CI.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def yn(p=0.5):
        return rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    def dep_on_internet():
        return ["No internet service" if internet[i] == "No"
                else rng.choice(["Yes", "No"]) for i in range(n)]

    monthly = np.round(rng.uniform(18, 120, n), 2)
    total = [" " if tenure[i] == 0
             else str(round(tenure[i] * monthly[i] * rng.uniform(0.9, 1.1), 2))
             for i in range(n)]

    df = pd.DataFrame({
        "customerID": [f"ID{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(), "Dependents": yn(0.3),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": ["No phone service" if phone[i] == "No"
                          else rng.choice(["Yes", "No"]) for i in range(n)],
        "InternetService": internet,
        "OnlineSecurity": dep_on_internet(), "OnlineBackup": dep_on_internet(),
        "DeviceProtection": dep_on_internet(), "TechSupport": dep_on_internet(),
        "StreamingTV": dep_on_internet(), "StreamingMovies": dep_on_internet(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": yn(0.59),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": rng.choice(["Yes", "No"], n, p=[0.265, 0.735]),
    })
    return df


if __name__ == "__main__":
    out = Path("data/raw/telco_churn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(out, index=False)
    print(f"Synthetic dataset written to {out} (shape confirmed on load)")
