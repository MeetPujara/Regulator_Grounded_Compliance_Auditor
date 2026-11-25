"""
Step 3: train two models (with and without protected attributes) and run group fairness tests.

Run from the project root:   python src/03_fairness_tests.py
Writes the full numbers to:  reports/fairness_results.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loading import load_south_german, load_taiwan
from models import THRESHOLD, feature_plan, fit_oof
from fairness import compare_groups, fairlearn_check

DATASETS = {
    "german": {"loader": load_south_german,
               "groups": {"age_group": "age_group",
                          "foreign_worker": "foreign_worker_label",
                          "personal_status_sex": "personal_status_sex_label"}},
    "taiwan": {"loader": load_taiwan,
               "groups": {"sex": "sex_label",
                          "marriage": "marriage_label",
                          "age_group": "age_group"}},
}
VARIANTS = {"uses_protected": True, "unaware": False}   # does the model see age/sex/etc.?
MODELS = ["logistic", "boosted"]


def choose_threshold(dataset, prob, truth):
    """German: fixed 0.5. Taiwan: pick the cut-off so overall approval equals the true
    share of good payers (about 78%), because 0.5 approves far too many people there."""
    if dataset == "taiwan":
        return float(np.quantile(prob, 1 - truth.mean()))
    return THRESHOLD


def worst_row(result):
    """The testable group with the lowest ratio (the one we should look at first)."""
    testable = result[result["ratio_vs_best"].notna() & (result["verdict"] != "reference group")]
    if testable.empty:
        return None
    return testable.sort_values("ratio_vs_best").iloc[0]


def main():
    all_results, matches, checks = [], 0, 0
    for name, cfg in DATASETS.items():
        df = cfg["loader"]()
        for variant, use_protected in VARIANTS.items():
            numeric, categorical = feature_plan(name, use_protected)
            for kind in MODELS:
                fitted = fit_oof(kind, df, numeric, categorical)
                cutoff = choose_threshold(name, fitted["prob"], df["approved"])
                df["decision"] = (fitted["prob"] >= cutoff).astype(int)
                auc = roc_auc_score(df["approved"], fitted["prob"])
                acc = accuracy_score(df["approved"], df["decision"])
                print(f"\n{name.upper()} | {variant} | {kind}:  AUC {auc:.3f}  "
                      f"accuracy {acc:.3f}  approval rate {df['decision'].mean():.3f}  cutoff {cutoff:.3f}")
                for attribute, column in cfg["groups"].items():
                    result = compare_groups(df, column)
                    top = result.loc[result["selection_rate"].idxmax(), "group"]
                    if result["testable"].all() and top == result["reference_group"].iloc[0]:
                        # cross-check vs Fairlearn (only valid when our benchmark is the top-rate group)
                        other = fairlearn_check(df, column)
                        if other is not None:
                            checks += 1
                            matches += int(abs(other - result["ratio_vs_best"].min()) < 1e-9)
                    row = worst_row(result)
                    if row is None:
                        print(f"   {attribute:<20} no testable comparison")
                    else:
                        p = "n/a" if pd.isna(row["p_value"]) else f"{row['p_value']:.4f}"
                        print(f"   {attribute:<20} lowest: {str(row['group'])[:34]:<34} n={row['n']:<6} "
                              f"ratio {row['ratio_vs_best']:.3f}  p={p}  -> {row['verdict']}")
                    result.insert(0, "attribute", attribute)
                    result.insert(0, "model", kind)
                    result.insert(0, "variant", variant)
                    result.insert(0, "dataset", name)
                    all_results.append(result)
    out = pd.concat(all_results, ignore_index=True)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/fairness_results.csv", index=False)
    print(f"\nSaved {len(out)} rows to reports/fairness_results.csv")
    if checks:
        print(f"Fairlearn cross-check: {matches} of {checks} smallest ratios matched.")


if __name__ == "__main__":
    main()