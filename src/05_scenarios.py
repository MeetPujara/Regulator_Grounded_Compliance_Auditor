"""
Step 5: run the clean vs planted-bias scenarios.
Run from the project root:   python src\05_scenarios.py

Writes: reports/scenario_fairness.csv
        reports/scenario_flips.csv
        reports/scenario_planted_truth.json

Reminder: Scenario A must stay quiet ABOUT THE SYNTHETIC COLUMNS. The real
German under-25 finding is separate and is expected to remain.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from data_loading import load_south_german
from fairness import compare_groups
from models import THRESHOLD, fit_oof
from scenarios import (TARGET_GROUP, TARGET_PROFILE, TARGET_ZONE,
                       add_synthetic_columns, plant_bias, run_flips,
                       save_truth, scenario_feature_plan)

MODELS = ["logistic", "boosted"]
VARIANTS = {"uses_protected": True, "unaware": False}
ATTRIBUTES = ["syn_pincode", "syn_surname_group", "syn_target_profile"]


def worst_row(result):
    """Lowest-ratio testable group (same rule as Step 3)."""
    t = result[result["ratio_vs_best"].notna() & (result["verdict"] != "reference group")]
    if t.empty:
        return None
    return t.sort_values("ratio_vs_best").iloc[0]


def main():
    base = add_synthetic_columns(load_south_german())
    clean = base.copy()
    planted, truth = plant_bias(base)
    scenarios = {"A_clean": clean, "B_planted": planted}
    clean.to_csv("reports/synthetic_scenario_A.csv", index=False)
    planted.to_csv("reports/synthetic_scenario_B.csv", index=False)

    print("SYNTHETIC columns added (fake pincode / surname group).")
    print(f"Planted in B: {truth['rows_changed']} approvals turned into rejections "
          f"(zone {truth['changed_in_target_zone']}, group {truth['changed_in_target_group']}, "
          f"both {truth['changed_in_both']}).")
    print(f"Approval rate {truth['approval_rate_before']:.3f} -> "
          f"{truth['approval_rate_after']:.3f}")

    fair_rows, flip_rows = [], []

    for sname, df in scenarios.items():
        population = ((df["syn_pincode"] == TARGET_ZONE)
                      & (df["syn_surname_group"] == TARGET_GROUP)).to_numpy()

        for variant, use_protected in VARIANTS.items():
            numeric, categorical = scenario_feature_plan(use_protected)
            for kind in MODELS:
                fitted = fit_oof(kind, df, numeric, categorical)
                cutoff = THRESHOLD                      # German rule, as in Step 3
                df["decision"] = (fitted["prob"] >= cutoff).astype(int)
                auc = roc_auc_score(df["approved"], fitted["prob"])

                print(f"\n{sname} | {variant} | {kind}:  AUC {auc:.3f}  "
                      f"approval {df['decision'].mean():.3f}")

                for attribute in ATTRIBUTES:
                    result = compare_groups(df, attribute)
                    row = worst_row(result)
                    if row is None:
                        print(f"   {attribute:<22} no testable comparison")
                    else:
                        p = "n/a" if pd.isna(row["p_value"]) else f"{row['p_value']:.4f}"
                        print(f"   {attribute:<22} lowest: {str(row['group'])[:24]:<24} "
                              f"n={row['n']:<5} ratio {row['ratio_vs_best']:.3f}  "
                              f"p={p}  -> {row['verdict']}")
                    result.insert(0, "attribute", attribute)
                    result.insert(0, "model", kind)
                    result.insert(0, "variant", variant)
                    result.insert(0, "scenario", sname)
                    fair_rows.append(result)

                flips = run_flips(df, fitted, cutoff, population)
                for _, r in flips.iterrows():
                    rate = "n/a" if pd.isna(r["flip_rate"]) else f"{r['flip_rate']:.3f}"
                    small = "  [small group]" if r["n_eligible"] < 30 else ""
                    print(f"   flip {r['flip']:<18} n={r['n_eligible']:<4} "
                          f"flips={r['n_flips']:<4} rate={rate}{small}")
                flips.insert(0, "model", kind)
                flips.insert(0, "variant", variant)
                flips.insert(0, "scenario", sname)
                flip_rows.append(flips)

    fair = pd.concat(fair_rows, ignore_index=True)
    flips = pd.concat(flip_rows, ignore_index=True)
    Path("reports").mkdir(exist_ok=True)
    fair.to_csv("reports/scenario_fairness.csv", index=False)
    flips.to_csv("reports/scenario_flips.csv", index=False)
    save_truth(truth)
    print(f"\nSaved {len(fair)} fairness rows and {len(flips)} flip rows to reports/")

    # ---- the two checks this step exists for -------------------------------
    def flags(scenario):
        t = fair[(fair["scenario"] == scenario) & (fair["variant"] == "uses_protected")]
        return t[t["verdict"].str.startswith("FLAG")]

    a_flags = flags("A_clean")
    b_flags = flags("B_planted")
    print("\nCHECK 1 - no crying wolf (Scenario A must have no synthetic FLAG): "
          + ("PASS" if a_flags.empty else "CHECK - investigate"))
    if not a_flags.empty:
        print(a_flags[["attribute", "model", "group", "ratio_vs_best", "p_value"]].to_string(index=False))
    print("CHECK 2 - planted bias detected (Scenario B must FLAG): "
          + ("PASS" if not b_flags.empty else "CHECK - not detected"))
    if not b_flags.empty:
        print(b_flags[["attribute", "model", "group", "ratio_vs_best", "p_value"]].to_string(index=False))

    unaware_flips = int(flips.loc[flips["variant"] == "unaware", "n_flips"].sum())
    print("CHECK 3 - unaware models show 0 flips: "
          + ("PASS" if unaware_flips == 0 else f"FAIL ({unaware_flips})"))

    print("\nNote: in Scenario B the outcome labels are themselves biased, so AUC")
    print("      is NOT a fairness measure there. That is why group tests exist.")


if __name__ == "__main__":
    main()