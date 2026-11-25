"""
Step 5 helper: SYNTHETIC India-style columns + two controlled scenarios.

Everything here is FAKE data added on top of the real German file. Column names
start with "syn_" so nothing synthetic can be mistaken for real applicant data.

Scenario A (clean)   : synthetic columns added, outcomes untouched.
Scenario B (planted) : some approvals turned into rejections for one pincode
                       zone and one surname group, so we know the exact truth.

The synthetic columns are drawn INDEPENDENTLY of the real features and of the
outcome. So in Scenario A there is no signal and no proxy by construction.
"""
import json

import numpy as np
import pandas as pd

from models import SEED, feature_plan

ZONES = ["SYN_PIN_1", "SYN_PIN_2", "SYN_PIN_3",
         "SYN_PIN_4", "SYN_PIN_5", "SYN_PIN_6"]
ZONE_P = [0.25, 0.20, 0.17, 0.15, 0.13, 0.10]

GROUPS = ["SYN_GROUP_A", "SYN_GROUP_B", "SYN_GROUP_C"]
GROUP_P_DEFAULT = [0.45, 0.30, 0.25]
GROUP_P_TARGET = [0.25, 0.60, 0.15]      # group B is more common in the target zone

TARGET_ZONE, REF_ZONE = "SYN_PIN_1", "SYN_PIN_2"
TARGET_GROUP, REF_GROUP = "SYN_GROUP_B", "SYN_GROUP_A"
TARGET_PROFILE = "target zone + group B"

SYN_COLUMNS = ["syn_pincode", "syn_surname_group"]

# Planted rejection rates (applied only to rows that were approved).
PLANT_ZONE, PLANT_GROUP, PLANT_BOTH_EXTRA = 0.30, 0.15, 0.20


def add_synthetic_columns(df, seed=SEED):
    """Attach the fake pincode / surname columns. Same seed -> same assignment."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    n = len(out)

    zone = rng.choice(ZONES, size=n, p=ZONE_P)
    group = np.empty(n, dtype=object)
    for z in ZONES:
        idx = np.flatnonzero(zone == z)
        p = GROUP_P_TARGET if z == TARGET_ZONE else GROUP_P_DEFAULT
        group[idx] = rng.choice(GROUPS, size=len(idx), p=p)

    out["syn_pincode"] = zone
    out["syn_surname_group"] = group
    out["syn_target_profile"] = np.where(
        (zone == TARGET_ZONE) & (group == TARGET_GROUP), TARGET_PROFILE, "everyone else")
    return out


def plant_bias(df, seed=SEED + 1):
    """Scenario B: turn some approvals into rejections. Returns (data, truth)."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    n = len(out)

    in_zone = out["syn_pincode"].to_numpy() == TARGET_ZONE
    in_group = out["syn_surname_group"].to_numpy() == TARGET_GROUP
    was_approved = out["approved"].to_numpy() == 1

    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    hit = ((in_zone & (u1 < PLANT_ZONE))
           | (in_group & (u2 < PLANT_GROUP))
           | (in_zone & in_group & (u3 < PLANT_BOTH_EXTRA)))
    flipped = hit & was_approved

    out["approved_before_plant"] = out["approved"]
    out.loc[flipped, "approved"] = 0

    truth = {
        "notice": "SYNTHETIC planted bias. Not real Indian applicant data.",
        "seed_columns": int(SEED), "seed_plant": int(seed),
        "target_zone": TARGET_ZONE, "reference_zone": REF_ZONE,
        "target_group": TARGET_GROUP, "reference_group": REF_GROUP,
        "plant_rates": {"zone_only": PLANT_ZONE, "group_only": PLANT_GROUP,
                        "both_extra": PLANT_BOTH_EXTRA},
        "rows_changed": int(flipped.sum()),
        "changed_in_target_zone": int((flipped & in_zone).sum()),
        "changed_in_target_group": int((flipped & in_group).sum()),
        "changed_in_both": int((flipped & in_zone & in_group).sum()),
        "approval_rate_before": float(df["approved"].mean()),
        "approval_rate_after": float(out["approved"].mean()),
    }
    return out, truth


def scenario_feature_plan(use_protected):
    """German features, plus the synthetic columns when the model may see them."""
    numeric, categorical = feature_plan("german", use_protected)
    if use_protected:
        categorical = categorical + SYN_COLUMNS
    return numeric, categorical


# All three tests run on the SAME people, so the rates compare fairly.
FLIP_SPECS = [
    {"name": "pincode only", "changes": {"syn_pincode": REF_ZONE}},
    {"name": "surname only", "changes": {"syn_surname_group": REF_GROUP}},
    {"name": "BUNDLE (both)", "changes": {"syn_pincode": REF_ZONE,
                                         "syn_surname_group": REF_GROUP}},
]


def run_flips(df, fitted, cutoff, population, specs=FLIP_SPECS):
    """Flip one or both attributes for rejected applicants in `population`.

    Always starts from the ORIGINAL row and uses the fold model that never saw
    that applicant. Returns one row per test.
    """
    prob, fold = fitted["prob"], fitted["fold"]
    models, features = fitted["models"], fitted["features"]

    idx = np.flatnonzero((prob < cutoff) & population)
    rows = []
    for spec in specs:
        row = {"flip": spec["name"], "n_eligible": len(idx),
               "n_flips": 0, "flip_rate": np.nan}
        if len(idx):
            changed = df.iloc[idx].copy()
            for col, val in spec["changes"].items():
                changed[col] = val
            X = changed[features]      # unaware: columns absent -> identical input
            new_prob = np.empty(len(idx))
            for k, model in enumerate(models):
                sel = fold[idx] == k
                if sel.any():
                    new_prob[sel] = model.predict_proba(X.iloc[sel])[:, 1]
            got = new_prob >= cutoff
            row["n_flips"] = int(got.sum())
            row["flip_rate"] = row["n_flips"] / len(idx)
        rows.append(row)
    return pd.DataFrame(rows)


def save_truth(truth, path="reports/scenario_planted_truth.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)