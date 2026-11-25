"""
Step 4a: run the flip tests on both datasets, both model kinds, both variants.
Run from the project root:   python src\04a_counterfactual_tests.py

Writes: reports/counterfactual_results.csv  (one row per flip)
        reports/counterfactual_flips.csv    (one row per applicant who flipped)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import load_south_german, load_taiwan
from models import THRESHOLD, feature_plan, fit_oof
from counterfactuals import FLIPS, run_counterfactuals

DATASETS = {"german": load_south_german, "taiwan": load_taiwan}
VARIANTS = {"uses_protected": True, "unaware": False}
MODELS = ["logistic", "boosted"]


def choose_threshold(dataset, prob, truth):
    # Same rule as Step 3. Copied here because 03_fairness_tests.py cannot be
    # imported (its filename starts with a digit).
    if dataset == "taiwan":
        return float(np.quantile(prob, 1 - truth.mean()))
    return THRESHOLD


def main():
    all_results, all_details = [], []

    for name, loader in DATASETS.items():
        df = loader()
        for variant, use_protected in VARIANTS.items():
            numeric, categorical = feature_plan(name, use_protected)
            for kind in MODELS:
                fitted = fit_oof(kind, df, numeric, categorical)  # same seed as Step 3
                cutoff = choose_threshold(name, fitted["prob"], df["approved"])
                results, details = run_counterfactuals(df, fitted, cutoff, FLIPS[name])

                print(f"\n{name.upper()} | {variant} | {kind}  (cutoff {cutoff:.3f})")
                for _, r in results.iterrows():
                    if r["n_rejected_eligible"] == 0:
                        print(f"   {r['flip']:<55} nobody eligible")
                    else:
                        small = "  [small group]" if r["n_rejected_eligible"] < 30 else ""
                        print(f"   {r['flip']:<55} n={r['n_rejected_eligible']:<5} "
                              f"flips={r['n_flips']:<4} rate={r['flip_rate']:.3f}{small}")

                results.insert(0, "model", kind)
                results.insert(0, "variant", variant)
                results.insert(0, "dataset", name)
                all_results.append(results)

                if len(details):
                    details.insert(0, "model", kind)
                    details.insert(0, "variant", variant)
                    details.insert(0, "dataset", name)
                    all_details.append(details)

    out = pd.concat(all_results, ignore_index=True)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/counterfactual_results.csv", index=False)
    det = (pd.concat(all_details, ignore_index=True) if all_details else
           pd.DataFrame(columns=["dataset", "variant", "model", "row",
                                 "flip", "old_prob", "new_prob"]))
    det.to_csv("reports/counterfactual_flips.csv", index=False)
    print(f"\nSaved {len(out)} rows to reports/counterfactual_results.csv")
    print(f"Saved {len(det)} flipped applicants to reports/counterfactual_flips.csv")

    # SANITY CHECK: unaware models cannot see the flipped columns, so they
    # must show ZERO flips. Anything else means a protected column leaked in.
    unaware_flips = int(out.loc[out["variant"] == "unaware", "n_flips"].sum())
    if unaware_flips != 0:
        raise AssertionError(
            f"SANITY CHECK FAILED: unaware models showed {unaware_flips} flips. "
            "A protected column leaked into the unaware features -- fix this "
            "before trusting any result.")
    print("SANITY CHECK PASSED: unaware models showed 0 flips, exactly as they must.")

    # Biggest flip rates among models that see the attribute (groups of 30+).
    big = out[(out["variant"] == "uses_protected") & (out["n_rejected_eligible"] >= 30)]
    big = big.sort_values("flip_rate", ascending=False).head(5)
    if len(big):
        print("\nLargest flip rates (uses_protected, groups of 30+):")
        for _, r in big.iterrows():
            print(f"   {r['dataset']:<7} {r['model']:<9} {r['flip']:<55} "
                  f"{r['n_flips']}/{r['n_rejected_eligible']} = {r['flip_rate']:.3f}")


if __name__ == "__main__":
    main()