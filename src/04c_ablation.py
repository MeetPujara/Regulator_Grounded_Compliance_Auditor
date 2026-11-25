"""
Step 4c: proxy ablation - do the leaking columns actually CARRY the gap?

Plain words: Step 4b showed that some columns can predict age. That does not
prove they cause the approval gap. Here we delete them and measure again.

Three feature sets, same models, same cutoff rule as Step 3:
  uses_protected : the model sees age
  unaware        : age deleted (the gap that survived)
  minus proxies  : age deleted AND the biggest leaking columns deleted

If the group's ratio climbs toward 1.0, those columns were carrying the gap.
We also report AUC, because deleting real credit features costs accuracy.

Run from the project root:   python src\04c_ablation.py
Writes: reports/ablation_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_loading import load_south_german, load_taiwan
from fairness import compare_groups
from models import THRESHOLD, feature_plan, fit_oof

LOADERS = {"german": load_south_german, "taiwan": load_taiwan}
MODELS = ["logistic", "boosted"]

# Leaking columns copied from reports/proxy_top_columns.csv (our Step 4b run).
# Hardcoded because that CSV does not carry a dataset column; the check below
# makes sure every name really exists in the unaware feature list.
ABLATIONS = [
    {
        "dataset": "german", "attribute": "age group",
        "group_column": "age_group", "focal_group": "under 25",
        "levels": {
            "minus top 3 proxies": ["housing", "employment_duration", "people_liable"],
        },
    },
    {
        "dataset": "taiwan", "attribute": "age group",
        "group_column": "age_group", "focal_group": "under 25",
        "levels": {
            "minus top 1 proxy": ["LIMIT_BAL"],
            "minus top 3 proxies": ["LIMIT_BAL", "EDUCATION", "BILL_AMT2"],
        },
    },
    {
        "dataset": "taiwan", "attribute": "sex",
        "group_column": "sex_label", "focal_group": "male",
        "levels": {
            "minus top 3 proxies": ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT6"],
        },
    },
]

_CACHE = {}


def get_df(name):
    """Load each dataset once (the Taiwan Excel read is slow)."""
    if name not in _CACHE:
        _CACHE[name] = LOADERS[name]()
    return _CACHE[name]


def choose_threshold(dataset, prob, truth):
    # Same rule as Step 3. Copied because 03_fairness_tests.py cannot be
    # imported (its filename starts with a digit).
    if dataset == "taiwan":
        return float(np.quantile(prob, 1 - truth.mean()))
    return THRESHOLD


def measure(df, dataset, kind, numeric, categorical, group_column, focal_group):
    """Train, score, and measure the focal group's gap for one feature set."""
    fitted = fit_oof(kind, df, numeric, categorical)
    cutoff = choose_threshold(dataset, fitted["prob"], df["approved"])
    scored = df.copy()
    scored["decision"] = (fitted["prob"] >= cutoff).astype(int)

    table = compare_groups(scored, group_column)
    hit = table[table["group"] == focal_group]
    if hit.empty:
        raise ValueError(f"Group '{focal_group}' not found in {group_column}")
    r = hit.iloc[0]

    return {
        "n_features": len(numeric) + len(categorical),
        "auc": roc_auc_score(df["approved"], fitted["prob"]),
        "approval_rate": float(scored["decision"].mean()),
        "group_n": int(r["n"]),
        "selection_rate": float(r["selection_rate"]),
        "ratio_vs_best": (float(r["ratio_vs_best"])
                          if pd.notna(r["ratio_vs_best"]) else np.nan),
        "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
        "verdict": r["verdict"],
    }


def main():
    rows = []

    for spec in ABLATIONS:
        name = spec["dataset"]
        df = get_df(name)
        num_p, cat_p = feature_plan(name, use_protected=True)
        num_u, cat_u = feature_plan(name, use_protected=False)

        configs = [("uses_protected", num_p, cat_p, ""),
                   ("unaware", num_u, cat_u, "")]
        for label, removed in spec["levels"].items():
            base = num_u + cat_u
            missing = [c for c in removed if c not in base]
            if missing:
                raise ValueError(f"{name}: {missing} are not in the unaware features")
            configs.append((label,
                            [c for c in num_u if c not in removed],
                            [c for c in cat_u if c not in removed],
                            ", ".join(removed)))

        for kind in MODELS:
            print(f"\n{name.upper()} | {spec['attribute']} | {kind}"
                  f"   (focal group: {spec['focal_group']})")
            seen = {}
            for label, num, cat, removed in configs:
                r = measure(df, name, kind, num, cat,
                            spec["group_column"], spec["focal_group"])
                seen[label] = r
                rows.append({"dataset": name, "attribute": spec["attribute"],
                             "model": kind, "feature_set": label,
                             "removed_columns": removed, **r})

                p = "n/a" if pd.isna(r["p_value"]) else f"{r['p_value']:.4f}"
                ratio = ("n/a" if pd.isna(r["ratio_vs_best"])
                         else f"{r['ratio_vs_best']:.3f}")
                print(f"   {label:<22} features={r['n_features']:<3} "
                      f"AUC {r['auc']:.3f}  n={r['group_n']:<5} ratio {ratio}  "
                      f"p={p}  -> {r['verdict']}")

            base = seen["unaware"]
            for label in spec["levels"]:
                r = seen[label]
                if pd.isna(base["ratio_vs_best"]) or pd.isna(r["ratio_vs_best"]):
                    continue
                drift = r["ratio_vs_best"] - base["ratio_vs_best"]
                cost = r["auc"] - base["auc"]
                print(f"      {label}: ratio {base['ratio_vs_best']:.3f} -> "
                      f"{r['ratio_vs_best']:.3f} ({drift:+.3f})   "
                      f"AUC cost {cost:+.3f}")

    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/ablation_results.csv", index=False)
    print(f"\nSaved {len(out)} rows to reports/ablation_results.csv")

    print("\nHow to read this:")
    print("  ratio moving toward 1.0 = those columns were carrying the gap")
    print("  ratio barely moving     = the gap lives somewhere else")
    print("  AUC cost                = the price. Housing, job tenure and credit")
    print("                            limit are real credit information, so this")
    print("                            is a diagnostic, NOT a recommended fix.")


if __name__ == "__main__":
    main()