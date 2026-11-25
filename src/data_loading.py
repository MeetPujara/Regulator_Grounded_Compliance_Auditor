"""
Step 2: load the datasets and DECODE them using the official UCI codetable.

Rules we follow:
  * South German Credit is our MAIN German file (UCI ships a corrected code table for it).
  * We NEVER create a 'sex' column for the German data. The code table says sex cannot
    be recovered (code 2 mixes female non-singles with male singles).
  * Sex tests use the Taiwan data, which has a clean SEX column.

Check the decoding by running:   python src/data_loading.py
"""
from pathlib import Path
import pandas as pd

# ---- Code tables copied from data/codetable.txt (South German Credit) -------------
SOUTH_GERMAN_CODES = {
    "status": {1: "no checking account", 2: "< 0 DM", 3: "0 to < 200 DM",
               4: ">= 200 DM / salary for at least 1 year"},
    "credit_history": {0: "delay in paying off in the past",
                       1: "critical account / other credits elsewhere",
                       2: "no credits taken / all credits paid back duly",
                       3: "existing credits paid back duly till now",
                       4: "all credits at this bank paid back duly"},
    "purpose": {0: "others", 1: "car (new)", 2: "car (used)", 3: "furniture/equipment",
                4: "radio/television", 5: "domestic appliances", 6: "repairs",
                7: "education", 8: "vacation", 9: "retraining", 10: "business"},
    "savings": {1: "unknown / no savings account", 2: "< 100 DM", 3: "100 to < 500 DM",
                4: "500 to < 1000 DM", 5: ">= 1000 DM"},
    "employment_duration": {1: "unemployed", 2: "< 1 yr", 3: "1 to < 4 yrs",
                            4: "4 to < 7 yrs", 5: ">= 7 yrs"},
    "installment_rate": {1: ">= 35", 2: "25 to < 35", 3: "20 to < 25", 4: "< 20"},
    "personal_status_sex": {1: "male: divorced/separated",
                            2: "female non-single OR male single (ambiguous)",
                            3: "male: married/widowed",
                            4: "female: single"},
    "other_debtors": {1: "none", 2: "co-applicant", 3: "guarantor"},
    "present_residence": {1: "< 1 yr", 2: "1 to < 4 yrs", 3: "4 to < 7 yrs", 4: ">= 7 yrs"},
    "property": {1: "unknown / no property", 2: "car or other",
                 3: "building society savings / life insurance", 4: "real estate"},
    "other_installment_plans": {1: "bank", 2: "stores", 3: "none"},
    "housing": {1: "for free", 2: "rent", 3: "own"},
    "number_credits": {1: "1", 2: "2-3", 3: "4-5", 4: ">= 6"},
    "job": {1: "unemployed / unskilled non-resident", 2: "unskilled resident",
            3: "skilled employee / official", 4: "manager / self-employed / highly qualified"},
    "people_liable": {1: "3 or more", 2: "0 to 2"},
    "telephone": {1: "no", 2: "yes (under customer name)"},
    "foreign_worker": {1: "yes", 2: "no"},
    "credit_risk": {0: "bad", 1: "good"},
}

SOUTH_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex", "other_debtors",
    "present_residence", "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk",
]


def _find(data_dir, patterns):
    for pattern in patterns:
        hits = sorted(Path(data_dir).rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {patterns} inside {data_dir}/")


def _age_group(age):
    return pd.Series(["under 25" if a < 25 else "25 and over" for a in age], index=age.index)


def load_south_german(data_dir="data"):
    """Return the decoded South German Credit table (1000 rows).

    Numeric columns keep their original codes (the model uses these).
    Each coded column also gets a '<name>_label' column (reports use these).
    """
    path = _find(data_dir, ["SouthGermanCredit.asc"])
    df = pd.read_csv(path, sep=r"\s+")
    if list(df.columns) != SOUTH_GERMAN_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    for column, mapping in SOUTH_GERMAN_CODES.items():
        unknown = set(df[column].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"{column}: codes {sorted(unknown)} are not in the code table")
        df[column + "_label"] = df[column].map(mapping)
    df["approved"] = (df["credit_risk"] == 1).astype(int)   # model approves "good" risks
    df["age_group"] = _age_group(df["age"])
    return df


def load_taiwan(data_dir="data"):
    """Return the Taiwan credit-card default table (30,000 rows) with a clean SEX label."""
    path = _find(data_dir, ["*default*credit*card*.xls*", "*default*credit*card*.csv"])
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=1)
        if "SEX" not in df.columns:
            df = pd.read_csv(path, header=0)
    else:
        df = None
        for header in (1, 0):
            for engine in ("calamine", None):
                try:
                    df = pd.read_excel(path, header=header, engine=engine)
                    break
                except (ImportError, ValueError):
                    continue
            if df is not None and "SEX" in df.columns:
                break
    if df is None or "SEX" not in df.columns:
        raise ValueError("Could not read the Taiwan file. Run 01_inspect_data.py and paste me the output.")
    df = df.rename(columns={"default payment next month": "default"})
    df["sex_label"] = df["SEX"].map({1: "male", 2: "female"})        # per UCI description
    df["marriage_label"] = df["MARRIAGE"].map(
        {1: "married", 2: "single", 3: "other", 0: "other (0 is undocumented)"})
    df["approved"] = (df["default"] == 0).astype(int)                # approve if no default
    df["age_group"] = _age_group(df["AGE"])
    return df


if __name__ == "__main__":
    sg = load_south_german()
    print("SOUTH GERMAN CREDIT:", sg.shape)
    for col in ["foreign_worker_label", "personal_status_sex_label", "age_group", "credit_risk_label"]:
        print(f"\n{col}")
        print(sg[col].value_counts().to_string())
    print("\nApproval rate (share of 'good'):", round(sg["approved"].mean(), 3))
    tw = load_taiwan()
    print("\n\nTAIWAN:", tw.shape)
    for col in ["sex_label", "marriage_label", "age_group"]:
        print(f"\n{col}")
        print(tw[col].value_counts().to_string())
    print("\nApproval rate (share with no default):", round(tw["approved"].mean(), 3))