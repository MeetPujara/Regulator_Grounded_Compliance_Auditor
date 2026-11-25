"""
Step 1: LOOK at the data before we trust it. Nothing is trained here.

Run from the project root (the folder that contains data/):
    python src/01_inspect_data.py
"""
from pathlib import Path
import pandas as pd

DATA = Path("data")

# Column NAMES for the Statlog file (german.data has no header row).
# We only name the columns here. We do NOT decode the A11/A92 style codes yet,
# because UCI says the Statlog code documentation has errors.
STATLOG_NAMES = [
    "status_checking", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker", "class",
]


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_file(patterns):
    """Look for a file anywhere inside data/ using simple name patterns."""
    for pattern in patterns:
        hits = sorted(DATA.rglob(pattern))
        if hits:
            return hits[0]
    return None


def read_excel_any(path, header):
    """Try the newer 'calamine' reader first, fall back to the default one."""
    for engine in ("calamine", None):
        try:
            return pd.read_excel(path, header=header, engine=engine)
        except (ImportError, ValueError):
            continue
    raise RuntimeError("Could not read the Excel file. Try: pip install python-calamine")


def check(ok, message):
    print(("  PASS   " if ok else "  CHECK  ") + message)


def basic_report(df):
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Missing values in total: {int(df.isna().sum().sum())}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())


def counts(df, column, title):
    print(f"\n{title}  [{column}]")
    print(df[column].value_counts(dropna=False).sort_index().to_string())


def age_summary(df, column):
    print(f"\nAge  [{column}]")
    print(df[column].describe().round(1).to_string())
    print(f"Share under 25: {(df[column] < 25).mean():.1%}")


def inspect_statlog():
    banner("1) Statlog German Credit  (german.data)")
    path = find_file(["german.data"])
    if path is None:
        print("NOT FOUND: put german.data inside the data/ folder.")
        return
    print(f"File: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=STATLOG_NAMES)
    basic_report(df)
    check(len(df) == 1000, "1000 rows expected")
    check(df.shape[1] == 21, "21 columns expected (20 features + class)")
    counts(df, "class", "Target (1 = good, 2 = bad)")
    check(df["class"].value_counts().to_dict() == {1: 700, 2: 300}, "700 good / 300 bad expected")
    counts(df, "personal_status_sex", "Personal status and sex (RAW codes, do not read as sex)")
    counts(df, "foreign_worker", "Foreign worker (RAW codes)")
    age_summary(df, "age")


def inspect_south_german():
    banner("2) South German Credit  (SouthGermanCredit.asc)")
    path = find_file(["SouthGermanCredit.asc"])
    if path is None:
        print("NOT FOUND: put SouthGermanCredit.asc inside the data/ folder.")
        return
    print(f"File: {path}")
    df = pd.read_csv(path, sep=r"\s+")
    basic_report(df)
    check(len(df) == 1000, "1000 rows expected")
    if "credit_risk" not in df.columns:
        print("  CHECK  Column 'credit_risk' not found. Paste the first 2 lines of the file to me.")
        print("Columns found:", list(df.columns))
        return
    counts(df, "credit_risk", "Target (credit_risk: 1 = good, 0 = bad)")
    counts(df, "personal_status_sex", "Personal status and sex (code 2 mixes male singles and female non-singles)")
    counts(df, "foreign_worker", "Foreign worker (codetable: 1 = yes, 2 = no)")
    age_summary(df, "age")
    codetable = find_file(["codetable.txt"])
    check(codetable is not None, "codetable.txt present in data/ (needed to decode the numbers)")


def inspect_taiwan():
    banner("3) Taiwan credit default  (Excel or CSV)")
    path = find_file(["*default*credit*card*.xls*", "*default*credit*card*.csv"])
    if path is None:
        print("NOT FOUND: put the 'default of credit card clients' file inside data/.")
        return
    print(f"File: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=1)
        if "SEX" not in df.columns:
            df = pd.read_csv(path, header=0)
    else:
        df = read_excel_any(path, header=1)   # row 1 is often a stray header row
        if "SEX" not in df.columns:
            df = read_excel_any(path, header=0)
    basic_report(df)
    check("SEX" in df.columns, "column SEX found (header row handled correctly)")
    check(len(df) == 30000, "30000 rows expected")
    if "SEX" not in df.columns:
        print("Columns found:", list(df.columns))
        return
    target = [c for c in df.columns if "default" in str(c).lower()]
    if target:
        counts(df, target[0], "Target")
    counts(df, "SEX", "Sex (1 = male, 2 = female per UCI)")
    if "MARRIAGE" in df.columns:
        counts(df, "MARRIAGE", "Marital status")
    if "AGE" in df.columns:
        age_summary(df, "AGE")


if __name__ == "__main__":
    if not DATA.exists():
        raise SystemExit("Run this from the project root, the folder that contains data/.")
    inspect_statlog()
    inspect_south_german()
    inspect_taiwan()
    banner("Reminder")
    print("Sex cannot be derived from the German files. Sex tests will use the Taiwan data.")