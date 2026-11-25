"""
Models (Step 3). We train two simple models and score EVERY applicant with a model
that never saw them (5-fold "cross-fitting"). That way the audit can look at all rows,
which matters because some groups are tiny (only 37 foreign workers in the German data).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
THRESHOLD = 0.5          # approve if predicted probability of "good" is at least this

# ---- German (South German Credit) ----------------------------------------------
GERMAN_NUMERIC = ["duration", "amount", "age"]
GERMAN_CATEGORICAL = [
    "status", "credit_history", "purpose", "savings", "employment_duration",
    "installment_rate", "personal_status_sex", "other_debtors", "present_residence",
    "property", "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker",
]
GERMAN_PROTECTED = ["age", "personal_status_sex", "foreign_worker"]

# ---- Taiwan ---------------------------------------------------------------------
TAIWAN_NUMERIC = (
    ["LIMIT_BAL", "AGE", "PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
)
TAIWAN_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
TAIWAN_PROTECTED = ["SEX", "MARRIAGE", "AGE"]     # EDUCATION stays: it is a proxy candidate

PLANS = {
    "german": (GERMAN_NUMERIC, GERMAN_CATEGORICAL, GERMAN_PROTECTED),
    "taiwan": (TAIWAN_NUMERIC, TAIWAN_CATEGORICAL, TAIWAN_PROTECTED),
}


def feature_plan(dataset, use_protected):
    """Return (numeric_columns, categorical_columns) for the model."""
    numeric, categorical, protected = PLANS[dataset]
    if not use_protected:
        numeric = [c for c in numeric if c not in protected]
        categorical = [c for c in categorical if c not in protected]
    return numeric, categorical


def make_model(kind, numeric, categorical):
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "logistic":
        prep = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", onehot, categorical)])
        classifier = LogisticRegression(max_iter=3000)
    elif kind == "boosted":
        prep = ColumnTransformer([("num", "passthrough", numeric), ("cat", onehot, categorical)])
        classifier = HistGradientBoostingClassifier(random_state=SEED)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("prep", prep), ("clf", classifier)])


def fit_oof(kind, df, numeric, categorical, target="approved", n_splits=5):
    """Cross-fit: every row gets a probability from a model trained WITHOUT that row.

    Returns the probabilities, which fold each row belonged to, and the fitted fold models
    (the counterfactual step will need them to re-score changed applicants).
    """
    X = df[numeric + categorical]
    y = df[target].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    prob = np.zeros(len(df))
    fold = np.zeros(len(df), dtype=int)
    models = []
    for k, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        model = make_model(kind, numeric, categorical).fit(X.iloc[train_idx], y[train_idx])
        prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        fold[test_idx] = k
        models.append(model)
    return {"prob": prob, "fold": fold, "models": models, "features": numeric + categorical}