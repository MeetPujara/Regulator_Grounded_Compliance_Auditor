"""
Step 4a helper: the "flip test" (counterfactuals).

Plain words: take an applicant the model REJECTED. Change exactly ONE thing
about them. Keep every other column identical. Ask the SAME fold model again
(the one that never saw this applicant in training). If the answer changes to
"approved", that one attribute alone changed the outcome.

Safety rules built in here:
- Always start from the ORIGINAL applicant (never chain flips).
- Only rejected applicants are tested (a flip = rejection -> approval).
- "unaware" models don't see the flipped columns, so they MUST show zero
  flips. The runner script checks this and fails loudly if not.
"""

import numpy as np
import pandas as pd

# Each flip = one allowed change. "from_value" = who it applies to,
# "to_value" = what we change it to. Nothing else in the row is touched.
# German personal-status codes (from data_loading.py / the real codetable):
#   1 = male divorced/separated, 2 = female non-single OR male single (ambiguous),
#   3 = male married/widowed,    4 = female single.
FLIPS = {
    "german": [
        {"name": "foreign worker: yes -> no",
         "column": "foreign_worker", "from_value": 1, "to_value": 2},
        {"name": "foreign worker: no -> yes",
         "column": "foreign_worker", "from_value": 2, "to_value": 1},
        {"name": "status: divorced/sep male -> married/widowed male",
         "column": "personal_status_sex", "from_value": 1, "to_value": 3},
        {"name": "status: married/widowed male -> divorced/sep male",
         "column": "personal_status_sex", "from_value": 3, "to_value": 1},
        {"name": "status: ambiguous group -> female single (AMBIGUOUS)",
         "column": "personal_status_sex", "from_value": 2, "to_value": 4},
        {"name": "status: female single -> ambiguous group (AMBIGUOUS)",
         "column": "personal_status_sex", "from_value": 4, "to_value": 2},
        {"name": "age: under 25 -> 25",
         "column": "age", "special": "under25", "to_value": 25},
    ],
    "taiwan": [
        {"name": "sex: male -> female",
         "column": "SEX", "from_value": 1, "to_value": 2},
        {"name": "sex: female -> male",
         "column": "SEX", "from_value": 2, "to_value": 1},
        {"name": "marriage: married -> single",
         "column": "MARRIAGE", "from_value": 1, "to_value": 2},
        {"name": "marriage: single -> married",
         "column": "MARRIAGE", "from_value": 2, "to_value": 1},
    ],
}


def _eligible(df, flip):
    """Which rows this flip applies to (before we look at rejections)."""
    values = df[flip["column"]].to_numpy()
    if flip.get("special") == "under25":
        return values < 25
    return values == flip["from_value"]


def run_counterfactuals(df, fitted, cutoff, flips):
    """Run all flips on one fitted model set.

    df      : the decoded dataset (all columns).
    fitted  : the dict returned by models.fit_oof (prob, fold, models, features).
    cutoff  : the approval cutoff used in Step 3.
    flips   : FLIPS[dataset_name].

    Returns (summary, details):
      summary : one row per flip  -> eligible count, flip count, flip rate.
      details : one row per applicant who actually flipped (for report examples).
    """
    prob = fitted["prob"]
    fold = fitted["fold"]
    models = fitted["models"]
    features = fitted["features"]

    decision = (prob >= cutoff).astype(int)
    rejected = decision == 0

    summary_rows, detail_rows = [], []

    for flip in flips:
        idx = np.flatnonzero(rejected & _eligible(df, flip))
        row = {"flip": flip["name"], "column": flip["column"],
               "n_rejected_eligible": len(idx), "n_flips": 0,
               "flip_rate": np.nan, "mean_prob_change": np.nan}

        if len(idx):
            flipped = df.iloc[idx].copy()                 # the ORIGINAL rows
            flipped[flip["column"]] = flip["to_value"]    # exactly one change
            X = flipped[features]  # unaware: column not in features -> identical input

            new_prob = np.empty(len(idx))
            for k, model in enumerate(models):            # same fold model per row
                sel = fold[idx] == k
                if sel.any():
                    new_prob[sel] = model.predict_proba(X.iloc[sel])[:, 1]

            became_approved = new_prob >= cutoff
            row["n_flips"] = int(became_approved.sum())
            row["flip_rate"] = row["n_flips"] / len(idx)
            row["mean_prob_change"] = float(np.mean(new_prob - prob[idx]))

            for j in np.flatnonzero(became_approved):
                detail_rows.append({"row": int(idx[j]), "flip": flip["name"],
                                    "old_prob": float(prob[idx[j]]),
                                    "new_prob": float(new_prob[j])})
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)