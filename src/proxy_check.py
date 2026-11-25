"""
Step 4b helper: the proxy check.

Plain words: can we GUESS a protected attribute (age group, sex, ...) using only
the columns the "unaware" model still sees? If yes, those columns are proxies:
removing the protected column did not remove the information.

Method: train a probe model (the same boosted model from Step 3) to predict the
protected attribute, with 5-fold cross-fitting so every row is predicted by a
probe that never saw it. Then shuffle each column in turn to see which ones the
probe was relying on (permutation importance).
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from models import SEED, make_model

# What we try to guess. "positive" present -> binary target (that value = 1).
# No "positive" -> multi-group target, keep the raw codes.
PROXY_TARGETS = {
    "german": [
        {"name": "age group (under 25 vs 25+)", "column": "age_group", "positive": "under 25"},
        {"name": "foreign worker (yes vs no)", "column": "foreign_worker", "positive": 1},
        {"name": "personal status (4 groups)", "column": "personal_status_sex"},
    ],
    "taiwan": [
        {"name": "sex (male vs female)", "column": "SEX", "positive": 1},
        {"name": "marriage (4 groups)", "column": "MARRIAGE"},
        {"name": "age group (under 25 vs 25+)", "column": "age_group", "positive": "under 25"},
    ],
}

# Our screening lines (NOT legal thresholds).
AUC_SOME, AUC_STRONG = 0.60, 0.70
LIFT_SOME, LIFT_STRONG = 0.07, 0.15   # balanced accuracy above chance, multi-group


def _verdict(binary, auc, lift):
    if binary:
        if auc >= AUC_STRONG:
            return "strong proxy signal"
        if auc >= AUC_SOME:
            return "some proxy signal"
        return "little/no proxy signal"
    if lift >= LIFT_STRONG:
        return "strong proxy signal"
    if lift >= LIFT_SOME:
        return "some proxy signal"
    return "little/no proxy signal"


def probe_target(df, numeric, categorical, spec, n_splits=5,
                 importance_rows=3000, n_repeats=5):
    """Try to predict one protected attribute from the non-protected columns.

    Returns (summary_row, top_columns_table).
    """
    col = spec["column"]
    if "positive" in spec:
        y = (df[col] == spec["positive"]).astype(int).to_numpy()
    else:
        y = df[col].to_numpy()

    features = numeric + categorical
    X = df[features]
    classes, counts = np.unique(y, return_counts=True)
    binary = len(classes) == 2
    minority_n = int(counts.min())
    chance = 0.5 if binary else 1.0 / len(classes)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_class = np.zeros(len(y), dtype=classes.dtype)
    oof_prob = np.zeros(len(y))
    first_fold = None

    for k, (tr, te) in enumerate(splitter.split(X, y)):
        probe = make_model("boosted", numeric, categorical).fit(X.iloc[tr], y[tr])
        p = probe.predict_proba(X.iloc[te])
        if binary:
            oof_prob[te] = p[:, 1]
            oof_class[te] = (p[:, 1] >= 0.5).astype(int)
        else:
            oof_class[te] = probe.classes_[np.argmax(p, axis=1)]
        if k == 0:
            first_fold = (probe, te)   # kept for the column-importance check

    ba = balanced_accuracy_score(y, oof_class)
    auc = roc_auc_score(y, oof_prob) if binary else np.nan
    lift = ba - chance

    # Which columns leak the most? Shuffle one column at a time on held-out rows
    # and measure how far the score falls. One fold, capped rows: indicative only.
    probe, te = first_fold
    rng = np.random.RandomState(SEED)
    take = rng.choice(te, size=min(importance_rows, len(te)), replace=False)
    scoring = "roc_auc" if binary else "balanced_accuracy"
    imp = permutation_importance(probe, X.iloc[take], y[take], scoring=scoring,
                                 n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    top = pd.DataFrame({"column": features,
                        "importance": imp.importances_mean,
                        "spread": imp.importances_std})
    top = top[top["importance"] > 0].sort_values("importance", ascending=False).head(5)
    top.insert(0, "target", spec["name"])

    if minority_n < 30:
        reliability = "unreliable: tiny group"
    elif minority_n < 100:
        reliability = "indicative only: small group"
    else:
        reliability = "ok"

    summary = {"target": spec["name"], "column": col, "n_rows": len(y),
               "n_groups": len(classes), "smallest_group_n": minority_n,
               "auc": auc, "balanced_accuracy": ba, "chance_level": chance,
               "lift_over_chance": lift,
               "verdict": _verdict(binary, auc if binary else 0.0, lift),
               "reliability": reliability}
    return summary, top