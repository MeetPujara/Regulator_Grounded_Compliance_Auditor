"""
Fairness tests (Step 3). Plain pandas + scipy, so every number can be checked by hand.

Important: the 0.8 line is a SCREENING heuristic borrowed from US employment guidelines.
It is not a legal verdict. The rule cards (Step 5) decide what it means in each country.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

MIN_GROUP = 30          # smaller groups get "insufficient data", never a number
MIN_REFERENCE = 100     # the benchmark group must be at least this big (tiny groups are too noisy)
SCREEN_RATIO = 0.8      # four-fifths screening line
PRACTICAL_RATIO = 0.9   # above this, a significant gap is too small to matter in practice
ALPHA = 0.05            # significance level (raw p-values, no multiple-comparison fix yet)


def _verdict(ratio, p_value):
    if ratio >= PRACTICAL_RATIO:
        return "no flag"
    if ratio >= SCREEN_RATIO:
        return "review: significant gap, ratio 0.8 to 0.9" if p_value < ALPHA else "no flag"
    return "FLAG: below 0.8 and significant" if p_value < ALPHA else "below 0.8 but not significant"


def compare_groups(df, group_col, decision_col="decision", truth_col="approved"):
    """Compare every group with the group that has the highest selection rate."""
    rows = []
    for group, sub in df.groupby(group_col):
        good = sub[sub[truth_col] == 1]
        rows.append({
            "group": group,
            "n": len(sub),
            "approved": int(sub[decision_col].sum()),
            "selection_rate": float(sub[decision_col].mean()),
            "n_truly_good": len(good),
            "tpr": float(good[decision_col].mean()) if len(good) else np.nan,
        })
    t = pd.DataFrame(rows)
    t["testable"] = t["n"] >= MIN_GROUP

    # equal opportunity: how often truly-good applicants are approved, vs the best group
    eligible = t["n_truly_good"] >= MIN_GROUP
    best_tpr = t.loc[eligible, "tpr"].max() if eligible.any() else np.nan
    t["tpr_gap"] = np.where(eligible, best_tpr - t["tpr"], np.nan)

    t["reference_group"] = None
    t["ratio_vs_best"] = np.nan
    t["p_value"] = np.nan
    t["verdict"] = "insufficient data (group smaller than %d)" % MIN_GROUP
    if not t["testable"].any():
        return t

    candidates = t[t["n"] >= MIN_REFERENCE]
    if candidates.empty:                      # every group is small: fall back to testable ones
        candidates = t[t["testable"]]
    ref = candidates.sort_values("selection_rate", ascending=False).iloc[0]
    t["reference_group"] = ref["group"]
    for i, r in t.iterrows():
        if not r["testable"]:
            continue
        if r["group"] == ref["group"]:
            t.loc[i, ["ratio_vs_best", "verdict"]] = [1.0, "reference group"]
            continue
        ratio = r["selection_rate"] / ref["selection_rate"] if ref["selection_rate"] > 0 else np.nan
        table = [[r["approved"], r["n"] - r["approved"]],
                 [ref["approved"], ref["n"] - ref["approved"]]]
        p_value = fisher_exact(table)[1]
        t.loc[i, "ratio_vs_best"] = ratio
        t.loc[i, "p_value"] = p_value
        t.loc[i, "verdict"] = _verdict(ratio, p_value)
    return t


def fairlearn_check(df, group_col, decision_col="decision", truth_col="approved"):
    """Independent cross-check of the smallest selection-rate ratio (None if fairlearn missing)."""
    try:
        from fairlearn.metrics import demographic_parity_ratio
    except ImportError:
        return None
    return float(demographic_parity_ratio(df[truth_col], df[decision_col],
                                          sensitive_features=df[group_col]))