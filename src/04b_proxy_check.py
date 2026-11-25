"""
Step 4b: run the proxy check on both datasets.
Run from the project root:   python src\04b_proxy_check.py

Writes: reports/proxy_results.csv      (one row per protected attribute)
        reports/proxy_top_columns.csv  (top 5 leaking columns per attribute)
"""
from pathlib import Path

import pandas as pd

from data_loading import load_south_german, load_taiwan
from models import feature_plan
from proxy_check import PROXY_TARGETS, probe_target

DATASETS = {"german": load_south_german, "taiwan": load_taiwan}


def main():
    summaries, tops = [], []

    for name, loader in DATASETS.items():
        df = loader()
        # Exactly the columns the "unaware" model still sees.
        numeric, categorical = feature_plan(name, use_protected=False)
        print(f"\n{name.upper()} - probing with {len(numeric) + len(categorical)} "
              f"non-protected columns")

        for spec in PROXY_TARGETS[name]:
            summary, top = probe_target(df, numeric, categorical, spec)
            summary.insert if False else None  # (no-op kept out; see below)
            summary["dataset"] = name
            summaries.append(summary)
            tops.append(top)

            if pd.isna(summary["auc"]):
                score = (f"balanced acc {summary['balanced_accuracy']:.3f} "
                         f"(chance {summary['chance_level']:.3f})")
            else:
                score = (f"AUC {summary['auc']:.3f}  balanced acc "
                         f"{summary['balanced_accuracy']:.3f}")
            print(f"\n   {summary['target']:<32} {score}")
            print(f"      -> {summary['verdict']}   [{summary['reliability']}]")
            if len(top):
                listed = ", ".join(f"{r['column']} ({r['importance']:.3f})"
                                   for _, r in top.iterrows())
                print(f"      biggest leaks: {listed}")
            else:
                print("      biggest leaks: none above zero")

    out = pd.DataFrame(summaries)
    cols = ["dataset", "target", "column", "n_rows", "n_groups", "smallest_group_n",
            "auc", "balanced_accuracy", "chance_level", "lift_over_chance",
            "verdict", "reliability"]
    out = out[cols]
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/proxy_results.csv", index=False)

    top_out = pd.concat(tops, ignore_index=True)
    top_out.to_csv("reports/proxy_top_columns.csv", index=False)
    print(f"\nSaved {len(out)} rows to reports/proxy_results.csv")
    print(f"Saved {len(top_out)} rows to reports/proxy_top_columns.csv")

    print("\nHow to read this:")
    print("  A high score means the protected information is STILL visible through")
    print("  other columns, so deleting the column did not delete the bias.")
    print("  That is why the German under-25 gap survived in Step 3 (0.83-0.89)")
    print("  and why the age flip in Step 4a was near zero.")
    print("  The 0.60 / 0.70 lines are our screening choice, not legal thresholds.")


if __name__ == "__main__":
    main()