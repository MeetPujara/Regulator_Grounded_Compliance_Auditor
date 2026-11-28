"""
RGCA — Regulatory-Grounded Compliance Auditor (Flask Web Application)
Replaces Streamlit with a production-style Flask + HTML/CSS/JS dashboard.
Run: python web_app.py
"""
import io
import json
import math
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
os.chdir(ROOT)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
load_dotenv(ROOT / ".env")

from applicability import check_applicability
from card_schema import load_cards
from data_loading import load_south_german, load_taiwan
from fairness import ALPHA, MIN_GROUP, PRACTICAL_RATIO, SCREEN_RATIO, compare_groups, fairlearn_check
from models import THRESHOLD, feature_plan, fit_oof
from retrieval import retrieve_cards
from scenarios import (
    TARGET_GROUP,
    TARGET_ZONE,
    add_synthetic_columns,
    plant_bias,
    scenario_feature_plan,
)
from verifier import verify_report

app = Flask(__name__)

# ── Configuration & Metadata ────────────────────────────────────────────
DEFAULT_AUDIT_DATE = "2026-11-24"

DATASETS = {
    "german": {"label": "South German Credit (UCI 522)", "short": "South German Credit", "rows": 1000},
    "taiwan": {"label": "Taiwan Credit Cards (UCI 353)", "short": "Taiwan Credit Cards", "rows": 30000},
    "scenario_B": {"label": "Scenario B (Planted Bias — Synthetic)", "short": "Scenario B (Planted)", "rows": 1000},
    "scenario_A": {"label": "Scenario A (Clean — Synthetic)", "short": "Scenario A (Clean)", "rows": 1000},
}

MODELS = {
    "logistic": "Logistic Regression",
    "boosted": "HistGradientBoosting",
}

GROUP_COLS = {
    "german": [
        ("age_group", "Age Group"),
        ("foreign_worker_label", "Foreign Worker"),
        ("personal_status_sex_label", "Marital Status / Sex"),
    ],
    "taiwan": [
        ("sex_label", "Sex"),
        ("marriage_label", "Marital Status"),
        ("age_group", "Age Group"),
    ],
    "scenario_B": [
        ("syn_pincode", "Pincode* (Synthetic)"),
        ("syn_surname_group", "Surname Group* (Synthetic)"),
        ("age_group", "Age Group"),
        ("foreign_worker_label", "Foreign Worker"),
    ],
    "scenario_A": [
        ("syn_pincode", "Pincode* (Synthetic)"),
        ("syn_surname_group", "Surname Group* (Synthetic)"),
        ("age_group", "Age Group"),
        ("foreign_worker_label", "Foreign Worker"),
    ],
}

READABLE_COLS = {
    "german": [
        ("age", "Age"),
        ("personal_status_sex_label", "Marital Status"),
        ("foreign_worker_label", "Foreign Worker"),
        ("duration", "Duration (mo)"),
        ("amount", "Amount"),
        ("housing_label", "Housing"),
        ("approved", "Actual"),
    ],
    "taiwan": [
        ("AGE", "Age"),
        ("sex_label", "Sex"),
        ("marriage_label", "Marital Status"),
        ("LIMIT_BAL", "Credit Limit"),
        ("default", "Defaulted?"),
        ("approved", "Actual"),
    ],
    "scenario_A": [
        ("age", "Age"),
        ("syn_pincode", "Pincode*"),
        ("syn_surname_group", "Surname*"),
        ("foreign_worker_label", "Foreign Worker"),
        ("amount", "Amount"),
        ("approved", "Actual"),
    ],
    "scenario_B": [
        ("age", "Age"),
        ("syn_pincode", "Pincode*"),
        ("syn_surname_group", "Surname*"),
        ("foreign_worker_label", "Foreign Worker"),
        ("amount", "Amount"),
        ("approved", "Actual"),
    ],
}

# ── In-Memory Caches & Session Activity ─────────────────────────────────
_DF_CACHE = {}
_FIT_CACHE = {}
_AUDIT_HISTORY = []  # Live session activity log (no fake records)


def _clean_val(v):
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    return str(v) if not isinstance(v, (str, int, float, bool)) else v


def load_df(key):
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset: {key}")
    if key not in _DF_CACHE:
        if key == "german":
            _DF_CACHE[key] = load_south_german()
        elif key == "taiwan":
            _DF_CACHE[key] = load_taiwan()
        elif key == "scenario_A":
            _DF_CACHE[key] = add_synthetic_columns(load_south_german())
        elif key == "scenario_B":
            planted, _ = plant_bias(add_synthetic_columns(load_south_german()))
            _DF_CACHE[key] = planted
    return _DF_CACHE[key]


def get_fitted(key, kind, use_protected=True):
    if kind not in MODELS:
        raise ValueError(f"Unknown model: {kind}")
    cache_key = (key, kind, bool(use_protected))
    if cache_key not in _FIT_CACHE:
        df = load_df(key)
        if key.startswith("scenario"):
            num, cat = scenario_feature_plan(use_protected)
        else:
            num, cat = feature_plan(key, use_protected)
        fitted = fit_oof(kind, df, num, cat)
        cutoff = (
            THRESHOLD
            if key != "taiwan"
            else float(np.quantile(fitted["prob"], 1 - df["approved"].mean()))
        )
        _FIT_CACHE[cache_key] = (fitted, cutoff, num + cat)
    return _FIT_CACHE[cache_key]


def get_valid_flips(key, row):
    """Single-attribute counterfactual flips valid for this specific applicant."""
    flips = []
    if key in ("german", "scenario_A", "scenario_B"):
        fw = row.get("foreign_worker")
        if fw in (1, 2):
            fw_opts = [
                {"value": 2, "label": "No (domestic worker)"},
                {"value": 1, "label": "Yes (foreign worker)"},
            ]
            other_fw = [o for o in fw_opts if o["value"] != fw]
            flips.append({
                "label": "Foreign Worker",
                "col": "foreign_worker",
                "old_val": int(fw),
                "new_val": other_fw[0]["value"],
                "old_display": "Yes (foreign worker)" if fw == 1 else "No (domestic worker)",
                "new_display": other_fw[0]["label"],
                "desc": "Foreign worker: yes → no" if fw == 1 else "Foreign worker: no → yes",
                "options": other_fw,
            })

        ps = row.get("personal_status_sex")
        ps_map = {
            1: "Male: divorced/separated",
            2: "Female non-single / Male single",
            3: "Male: married/widowed",
            4: "Female: single",
        }
        if ps in ps_map:
            other_ps = [{"value": k, "label": v} for k, v in ps_map.items() if k != ps]
            default_ps = next((o for o in other_ps if o["value"] in (1, 3)), other_ps[0])
            flips.append({
                "label": "Marital Status",
                "col": "personal_status_sex",
                "old_val": int(ps),
                "new_val": default_ps["value"],
                "old_display": ps_map[ps],
                "new_display": default_ps["label"],
                "desc": f"Marital status: {ps_map[ps]} → {default_ps['label']}",
                "options": other_ps,
            })

        age_val = int(row.get("age", 35))
        age_targets = [a for a in (22, 30, 35, 45, 60) if a != age_val]
        default_age = 35 if age_val < 25 else 22
        flips.append({
            "label": "Age",
            "col": "age",
            "old_val": age_val,
            "new_val": default_age,
            "old_display": f"{age_val} years",
            "new_display": f"{default_age} years",
            "desc": f"Age: {age_val} → {default_age}",
            "options": [{"value": a, "label": f"{a} years"} for a in age_targets],
        })

        if key.startswith("scenario"):
            cur_pin = row.get("syn_pincode")
            pin_zones = ["SYN_PIN_1", "SYN_PIN_2", "SYN_PIN_3", "SYN_PIN_4", "SYN_PIN_5", "SYN_PIN_6"]
            other_pins = [{"value": z, "label": z} for z in pin_zones if z != cur_pin]
            if cur_pin in pin_zones:
                default_pin = "SYN_PIN_2" if cur_pin != "SYN_PIN_2" else "SYN_PIN_1"
                flips.append({
                    "label": "Pincode* (Synthetic)",
                    "col": "syn_pincode",
                    "old_val": cur_pin,
                    "new_val": default_pin,
                    "old_display": cur_pin,
                    "new_display": default_pin,
                    "desc": f"Pincode*: {cur_pin} → {default_pin}",
                    "options": other_pins,
                })
            cur_grp = row.get("syn_surname_group")
            grp_list = ["SYN_GROUP_A", "SYN_GROUP_B", "SYN_GROUP_C"]
            other_grps = [{"value": g, "label": g} for g in grp_list if g != cur_grp]
            if cur_grp in grp_list:
                default_grp = "SYN_GROUP_A" if cur_grp != "SYN_GROUP_A" else "SYN_GROUP_B"
                flips.append({
                    "label": "Surname Group* (Synthetic)",
                    "col": "syn_surname_group",
                    "old_val": cur_grp,
                    "new_val": default_grp,
                    "old_display": cur_grp,
                    "new_display": default_grp,
                    "desc": f"Surname*: {cur_grp} → {default_grp}",
                    "options": other_grps,
                })
    elif key == "taiwan":
        sex = row.get("SEX")
        if sex in (1, 2):
            other_sex = [{"value": 2, "label": "Female (2)"}] if sex == 1 else [{"value": 1, "label": "Male (1)"}]
            flips.append({
                "label": "Sex",
                "col": "SEX",
                "old_val": int(sex),
                "new_val": other_sex[0]["value"],
                "old_display": "Male (1)" if sex == 1 else "Female (2)",
                "new_display": other_sex[0]["label"],
                "desc": "Sex: male → female" if sex == 1 else "Sex: female → male",
                "options": other_sex,
            })

        mar = row.get("MARRIAGE")
        mar_map = {1: "Married (1)", 2: "Single (2)", 3: "Others (3)"}
        if mar in mar_map:
            other_mar = [{"value": k, "label": v} for k, v in mar_map.items() if k != mar]
            flips.append({
                "label": "Marital Status",
                "col": "MARRIAGE",
                "old_val": int(mar),
                "new_val": other_mar[0]["value"],
                "old_display": mar_map[mar],
                "new_display": other_mar[0]["label"],
                "desc": f"Marriage: {mar_map[mar]} → {other_mar[0]['label']}",
                "options": other_mar,
            })

        age_val = int(row.get("AGE", 35))
        age_targets = [a for a in (22, 30, 35, 45, 60) if a != age_val]
        default_age = 35 if age_val < 25 else 22
        flips.append({
            "label": "Age",
            "col": "AGE",
            "old_val": age_val,
            "new_val": default_age,
            "old_display": f"{age_val} years",
            "new_display": f"{default_age} years",
            "desc": f"Age: {age_val} → {default_age}",
            "options": [{"value": a, "label": f"{a} years"} for a in age_targets],
        })
    return flips


def log_activity(action, detail, status="info"):
    _AUDIT_HISTORY.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail,
        "status": status,
    })
    if len(_AUDIT_HISTORY) > 25:
        _AUDIT_HISTORY.pop()


def _system_status():
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    reports_dir = ROOT / "reports"
    report_files = list(reports_dir.glob("*")) if reports_dir.exists() else []
    cards = load_cards()
    return {
        "groq_configured": groq_configured,
        "cards_count": len(cards),
        "datasets_count": 3,  # South German, Taiwan, Synthetic India Scenarios
        "models_count": len(MODELS),
        "report_files_count": len(report_files),
        "audit_date": DEFAULT_AUDIT_DATE,
    }


def _compute_model_metrics(ds_key, kind, use_protected=True):
    """Compute actual out-of-fold classification metrics without retraining."""
    fitted, cutoff, _ = get_fitted(ds_key, kind, use_protected)
    df = load_df(ds_key)
    y_true = df["approved"].to_numpy()
    prob = fitted["prob"]
    y_pred = (prob >= cutoff).astype(int)
    return {
        "total": int(len(y_true)),
        "approved": int(y_pred.sum()),
        "rejected": int((1 - y_pred).sum()),
        "approval_rate": round(float(y_pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "cutoff": round(float(cutoff), 4),
    }


def _build_audit_payload(ds_key="german", kind="logistic", jurisdiction="US", audit_date=DEFAULT_AUDIT_DATE):
    """Assemble a complete, grounded audit report payload from actual data."""
    df = load_df(ds_key).copy()
    perf_prot = _compute_model_metrics(ds_key, kind, True)
    perf_unaware = _compute_model_metrics(ds_key, kind, False)

    fitted, cutoff, _ = get_fitted(ds_key, kind, True)
    df["decision"] = (fitted["prob"] >= cutoff).astype(int)

    fairness_rows = []
    for col, label in GROUP_COLS.get(ds_key, []):
        if col not in df.columns:
            continue
        tbl = compare_groups(df, col, "decision", "approved")
        for _, r in tbl.iterrows():
            fairness_rows.append({
                "attribute": label,
                "group": str(r["group"]),
                "n": int(r["n"]),
                "approved": int(r["approved"]),
                "rejected": int(r["n"] - r["approved"]),
                "selection_rate": round(float(r["selection_rate"]), 4),
                "ratio_vs_best": _clean_val(r["ratio_vs_best"]),
                "p_value": _clean_val(r["p_value"]),
                "tpr_gap": _clean_val(r["tpr_gap"]),
                "reference_group": _clean_val(r["reference_group"]),
                "verdict": str(r["verdict"]),
            })

    cf_csv = ROOT / "reports" / "counterfactual_flips.csv"
    cf_rows = []
    if cf_csv.exists():
        cdf = pd.read_csv(cf_csv)
        cdf = cdf[(cdf["dataset"] == ds_key) & (cdf["model"] == kind)]
        cf_rows = [{k: _clean_val(v) for k, v in r.items()} for r in cdf.to_dict(orient="records")]

    proxy_csv = ROOT / "reports" / "proxy_results.csv"
    proxy_rows = []
    if proxy_csv.exists():
        pdf = pd.read_csv(proxy_csv)
        pdf = pdf[pdf["dataset"] == ds_key]
        proxy_rows = [{k: _clean_val(v) for k, v in r.items()} for r in pdf.to_dict(orient="records")]

    cards = load_cards()
    legal_results = []
    for c in cards:
        if c["jurisdiction"] == jurisdiction:
            lab, reason = check_applicability(c, jurisdiction, "lending", audit_date)
            legal_results.append({
                "id": c["id"],
                "clause": c["clause"],
                "effective_from": c.get("effective_from"),
                "status": c["status"],
                "label": lab,
                "reason": reason,
                "obligation": c["obligation"],
            })

    flags = [r for r in fairness_rows if r["verdict"].startswith("FLAG") or r["verdict"].startswith("review")]
    return {
        "project": "RGCA – Regulatory-Grounded Compliance Auditor",
        "audit_id": f"RGCA-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_date": audit_date,
        "dataset": {"key": ds_key, "label": DATASETS[ds_key]["label"], "rows": len(df)},
        "model": {"key": kind, "label": MODELS[kind]},
        "performance": {"uses_protected": perf_prot, "unaware": perf_unaware},
        "group_fairness": fairness_rows,
        "counterfactual_findings": cf_rows if cf_rows else "Not evaluated for this dataset/model in precomputed CSV",
        "proxy_findings": proxy_rows if proxy_rows else "Not evaluated for this dataset",
        "explainability": (
            "Linear standardized coefficients available for Logistic Regression via /api/explainability"
            if kind == "logistic"
            else "Not evaluated: HistGradientBoosting does not expose linear coefficients without external SHAP"
        ),
        "legal_jurisdiction": jurisdiction,
        "legal_applicability": legal_results,
        "warnings_and_limitations": [
            "The 0.80 selection-rate ratio is a screening heuristic borrowed from US employment guidelines, not a universal legal standard.",
            "Groups with fewer than 30 applicants are marked insufficient data due to statistical instability.",
            "Proxy associations indicate statistical predictability, not proven causal discrimination.",
            "This report is for compliance auditing support and does not constitute definitive legal advice.",
        ],
        "recommendations": [
            f"Investigate {len(flags)} flagged/review group disparities before model deployment." if flags else "No group-level screening flags triggered; continue periodic monitoring.",
            "Review proxy feature leakage if deploying an unaware model without protected attributes.",
            "Verify any 'Needs human review' or 'Not yet in force' legal rule cards with qualified counsel.",
        ],
        "summary": {
            "total_applicants": len(df),
            "approved_count": perf_prot["approved"],
            "rejected_count": perf_prot["rejected"],
            "fairness_flags_or_reviews": len(flags),
            "applicable_legal_rules": sum(1 for r in legal_results if r["label"] == "Applies"),
        },
    }


# ── Page Routes ─────────────────────────────────────────────────────────
def _overview_dataset_payload(dataset_key):
    """Compute overview KPIs, model metrics, and chart groups for a dataset."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    perf = _compute_model_metrics(dataset_key, "logistic", True)
    fitted, cutoff, _ = get_fitted(dataset_key, "logistic", True)
    df = load_df(dataset_key).copy()
    df["decision"] = (fitted["prob"] >= cutoff).astype(int)
    chart_groups = []
    for col, label in GROUP_COLS[dataset_key]:
        tbl = compare_groups(df, col, "decision", "approved")
        for _, r in tbl.iterrows():
            chart_groups.append({
                "attribute": label,
                "group": str(r["group"]),
                "rate": round(float(r["selection_rate"]) * 100, 1),
                "ratio": _clean_val(r["ratio_vs_best"]),
                "n": int(r["n"]),
                "verdict": str(r["verdict"]),
            })
    return {
        "dataset": dataset_key,
        "dataset_short": DATASETS[dataset_key]["short"],
        "perf": perf,
        "chart_groups": chart_groups,
    }


@app.route("/api/overview")
def api_overview():
    dataset = request.args.get("dataset", "german")
    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400
    return jsonify(_overview_dataset_payload(dataset))


@app.route("/")
def page_overview():
    cards = load_cards()
    fair_csv = ROOT / "reports" / "fairness_results.csv"
    cf_csv = ROOT / "reports" / "counterfactual_flips.csv"

    protected_groups_count = 0
    fairness_tests_count = 0
    fairness_warnings_count = 0
    if fair_csv.exists():
        fdf = pd.read_csv(fair_csv)
        protected_groups_count = int(fdf["group"].nunique())
        fairness_tests_count = int(len(fdf))
        fairness_warnings_count = int(
            fdf["verdict"].astype(str).str.startswith(("FLAG", "review")).sum()
        )

    counterfactual_tests_count = int(len(pd.read_csv(cf_csv))) if cf_csv.exists() else 0

    applicable_rules_count = sum(
        1 for c in cards
        if check_applicability(c, c["jurisdiction"], "lending", DEFAULT_AUDIT_DATE)[0] == "Applies"
    )

    reports_dir = ROOT / "reports"
    report_files = sorted([p.name for p in reports_dir.glob("*") if p.is_file()]) if reports_dir.exists() else []

    # Real fairness overview chart data (German + Logistic by default)
    ov = _overview_dataset_payload("german")
    perf_german = ov["perf"]
    chart_groups = ov["chart_groups"]

    components = [
        {"name": "Machine Learning", "module": "src/models.py", "status": "implemented", "detail": "5-fold cross-fitting (Logistic Regression & HistGradientBoosting)"},
        {"name": "Fairness Engine", "module": "src/fairness.py", "status": "implemented", "detail": "Selection-rate ratio, Fisher exact p-value, Equal Opportunity gap"},
        {"name": "Legal Retrieval", "module": "src/retrieval.py", "status": "implemented", "detail": "Hybrid BM25 + MiniLM/FAISS over 18 verified rule cards"},
        {"name": "Applicability Checker", "module": "src/applicability.py", "status": "implemented", "detail": "Deterministic jurisdiction, topic, date & status rules"},
        {
            "name": "LLM Explanation",
            "module": "src/report_writer.py",
            "status": "implemented" if os.getenv("GROQ_API_KEY") else "unavailable",
            "detail": "Groq LLM prose generator (active)" if os.getenv("GROQ_API_KEY") else "GROQ_API_KEY not set in .env (fallback/disabled)",
        },
        {"name": "Citation Verifier", "module": "src/verifier.py", "status": "implemented", "detail": "Card ID, label-respect & source-file keyword verification"},
    ]

    return render_template(
        "overview.html",
        active_page="overview",
        page_title="Overview",
        breadcrumbs=["RGCA", "Overview"],
        sys_status=_system_status(),
        metrics={
            "datasets": 3,
            "models": 2,
            "protected_groups": protected_groups_count,
            "rule_cards": len(cards),
            "applicable_rules": applicable_rules_count,
            "report_artifacts": len(report_files),
            "total_applicants": perf_german["total"],
            "total_approved": perf_german["approved"],
            "total_rejected": perf_german["rejected"],
            "fairness_tests": fairness_tests_count,
            "counterfactual_tests": counterfactual_tests_count,
            "fairness_warnings": fairness_warnings_count,
        },
        perf_german=perf_german,
        chart_groups=chart_groups,
        components=components,
        history=_AUDIT_HISTORY,
        report_files=report_files,
    )


@app.route("/live-audit")
def page_live_audit():
    return render_template(
        "live_audit.html",
        active_page="live_audit",
        page_title="Live Applicant Audit",
        breadcrumbs=["RGCA", "Live Audit"],
        sys_status=_system_status(),
        datasets=DATASETS,
        models=MODELS,
    )


@app.route("/group-fairness")
def page_group_fairness():
    return render_template(
        "group_fairness.html",
        active_page="group_fairness",
        page_title="Group Fairness Analysis",
        breadcrumbs=["RGCA", "Group Fairness"],
        sys_status=_system_status(),
        datasets=DATASETS,
        models=MODELS,
    )


@app.route("/counterfactuals")
def page_counterfactuals():
    cf_csv = ROOT / "reports" / "counterfactual_flips.csv"
    summary_rows = []
    if cf_csv.exists():
        cdf = pd.read_csv(cf_csv)
        summary_rows = [
            {k: _clean_val(v) for k, v in row.items()}
            for row in cdf.to_dict(orient="records")
        ]
    return render_template(
        "counterfactuals.html",
        active_page="counterfactuals",
        page_title="Counterfactual Fairness Lab",
        breadcrumbs=["RGCA", "Counterfactual Testing"],
        sys_status=_system_status(),
        datasets=DATASETS,
        models=MODELS,
        summary_rows=summary_rows,
    )


@app.route("/legal")
def page_legal():
    fair_csv = ROOT / "reports" / "fairness_results.csv"
    worst_findings = []
    if fair_csv.exists():
        fdf = pd.read_csv(fair_csv)
        fdf = fdf[(fdf["variant"] == "uses_protected") & (fdf["ratio_vs_best"].notna())]
        worst = fdf.sort_values("ratio_vs_best").head(6)
        worst_findings = [{k: _clean_val(v) for k, v in r.items()} for r in worst.to_dict(orient="records")]

    return render_template(
        "legal.html",
        active_page="legal",
        page_title="Legal Intelligence",
        breadcrumbs=["RGCA", "Legal Intelligence"],
        sys_status=_system_status(),
        worst_findings=worst_findings,
        default_date=DEFAULT_AUDIT_DATE,
    )


@app.route("/scenarios")
def page_scenarios():
    return render_template(
        "scenarios.html",
        active_page="scenarios",
        page_title="Scenario Testing",
        breadcrumbs=["RGCA", "Scenario Testing"],
        sys_status=_system_status(),
    )


@app.route("/reports")
def page_reports():
    reports_dir = ROOT / "reports"
    files = []
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*")):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "ext": p.suffix.lower().lstrip("."),
                })
    return render_template(
        "reports.html",
        active_page="reports",
        page_title="Audit Reports",
        breadcrumbs=["RGCA", "Audit Reports"],
        sys_status=_system_status(),
        report_files=files,
    )


@app.route("/settings")
def page_settings():
    return render_template(
        "settings.html",
        active_page="settings",
        page_title="Settings",
        breadcrumbs=["RGCA", "Settings"],
        sys_status=_system_status(),
        thresholds={
            "screen_ratio": SCREEN_RATIO,
            "practical_ratio": PRACTICAL_RATIO,
            "min_group": MIN_GROUP,
            "alpha": ALPHA,
            "model_threshold": THRESHOLD,
        },
    )


# ── API Routes ──────────────────────────────────────────────────────────
@app.route("/api/applicants")
def api_applicants():
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    use_prot = request.args.get("use_protected", "true").lower() == "true"
    search = request.args.get("search", "").strip().lower()
    decision_filter = request.args.get("decision", "all")
    sort_col = request.args.get("sort", "id")
    sort_dir = request.args.get("dir", "asc")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(5, int(request.args.get("per_page", 15))))

    fitted, cutoff, _ = get_fitted(ds_key, kind, use_prot)
    df = load_df(ds_key).copy()
    df["_id"] = df.index
    df["_prob"] = fitted["prob"]
    df["_decision"] = (df["_prob"] >= cutoff).astype(int)

    if decision_filter == "approved":
        df = df[df["_decision"] == 1]
    elif decision_filter == "rejected":
        df = df[df["_decision"] == 0]

    cols_meta = [(c, lbl) for c, lbl in READABLE_COLS[ds_key] if c in df.columns]

    if search:
        search_cols = ["_id"] + [c for c, _ in cols_meta]
        mask = pd.Series(False, index=df.index)
        for c in search_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(search, regex=False, na=False)
        df = df[mask]

    # Sorting
    sort_map = {"id": "_id", "prob": "_prob", "decision": "_decision"}
    actual_sort = sort_map.get(sort_col, sort_col if sort_col in df.columns else "_id")
    df = df.sort_values(actual_sort, ascending=(sort_dir == "asc"))

    total = len(df)
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)
    start = (page - 1) * per_page
    slice_df = df.iloc[start : start + per_page]

    rows = []
    for _, r in slice_df.iterrows():
        cells = {c: _clean_val(r[c]) for c, _ in cols_meta}
        rows.append({
            "id": int(r["_id"]),
            "cells": cells,
            "prob": round(float(r["_prob"]), 4),
            "decision": int(r["_decision"]),
            "actual": int(r["approved"]),
        })

    return jsonify({
        "columns": [{"key": c, "label": lbl} for c, lbl in cols_meta],
        "rows": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "cutoff": round(float(cutoff), 4),
        "is_synthetic": ds_key.startswith("scenario"),
    })


@app.route("/api/applicant/<int:row_idx>")
def api_applicant_detail(row_idx):
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    use_prot = request.args.get("use_protected", "true").lower() == "true"

    fitted, cutoff, features = get_fitted(ds_key, kind, use_prot)
    df = load_df(ds_key)
    if row_idx < 0 or row_idx >= len(df):
        return jsonify({"error": "Applicant index out of range"}), 404

    row = df.iloc[row_idx]
    prob = float(fitted["prob"][row_idx])
    decision = int(prob >= cutoff)
    flips = get_valid_flips(ds_key, row)

    # Display readable attributes
    attrs = {}
    for c, lbl in READABLE_COLS[ds_key]:
        if c in df.columns:
            attrs[lbl] = _clean_val(row[c])

    # Include model feature values for transparency
    feature_values = {f: _clean_val(row[f]) for f in features if f in df.columns}

    return jsonify({
        "id": row_idx,
        "prob": round(prob, 4),
        "decision": decision,
        "cutoff": round(float(cutoff), 4),
        "attributes": attrs,
        "feature_values": feature_values,
        "flips": flips,
    })


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    data = request.get_json(force=True)
    ds_key = data.get("dataset", "german")
    kind = data.get("model", "logistic")
    use_prot = bool(data.get("use_protected", True))
    row_idx = int(data.get("row_idx", 0))
    col = data.get("col")

    fitted, cutoff, features = get_fitted(ds_key, kind, use_prot)
    df = load_df(ds_key)
    if row_idx < 0 or row_idx >= len(df):
        return jsonify({"error": "Applicant index out of range"}), 404

    row = df.iloc[row_idx]
    valid_flips = get_valid_flips(ds_key, row)
    chosen = next((f for f in valid_flips if f["col"] == col), None)
    if not chosen:
        return jsonify({"error": f"Invalid or unsupported single-attribute flip for column '{col}'"}), 400

    req_new_val = data.get("new_val")
    target_val = chosen["new_val"]
    target_display = chosen["new_display"]
    if req_new_val is not None:
        matched_opt = next((o for o in chosen.get("options", []) if str(o["value"]) == str(req_new_val)), None)
        if not matched_opt:
            return jsonify({"error": f"Invalid target value '{req_new_val}' for '{col}'"}), 400
        target_val = matched_opt["value"]
        target_display = matched_opt["label"]

    chosen_desc = f"{chosen['label']}: {chosen['old_display']} → {target_display}"
    chosen_out = {**chosen, "new_val": target_val, "new_display": target_display, "desc": chosen_desc}

    changed = row.copy()
    changed[col] = target_val
    X = pd.DataFrame([changed[features]])
    fold_model = fitted["models"][fitted["fold"][row_idx]]

    old_prob = float(fitted["prob"][row_idx])
    new_prob = float(fold_model.predict_proba(X)[0, 1])
    old_decision = int(old_prob >= cutoff)
    new_decision = int(new_prob >= cutoff)
    decision_changed = old_decision != new_decision

    if decision_changed:
        explanation = (
            f"The model changed its decision after modifying {chosen['label']}. "
            f"All other applicant characteristics remained unchanged. "
            f"This indicates a potential fairness concern requiring further investigation."
        )
    else:
        explanation = (
            f"The model decision remained unchanged after modifying {chosen['label']}. "
            f"No decision flip was observed in this test."
        )

    finding_text = (
        f"Changing only {chosen_desc} for applicant #{row_idx} in {DATASETS[ds_key]['short']} "
        f"({MODELS[kind]}) changed probability from {old_prob:.3f} to {new_prob:.3f} "
        f"(decision changed: {'YES' if decision_changed else 'NO'})."
    )

    log_activity(
        "Counterfactual Test",
        f"{DATASETS[ds_key]['short']} · Row #{row_idx} · {chosen_desc} → {'CHANGED' if decision_changed else 'UNCHANGED'}",
        status="warning" if decision_changed else "success",
    )

    return jsonify({
        "row_idx": row_idx,
        "flip": chosen_out,
        "old_prob": round(old_prob, 4),
        "new_prob": round(new_prob, 4),
        "prob_diff": round(new_prob - old_prob, 4),
        "cutoff": round(float(cutoff), 4),
        "old_decision": old_decision,
        "new_decision": new_decision,
        "decision_changed": decision_changed,
        "explanation": explanation,
        "finding_text": finding_text,
    })


@app.route("/api/fairness")
def api_fairness():
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    use_prot = request.args.get("use_protected", "true").lower() == "true"
    selected_attr = request.args.get("attribute", "all")

    fitted, cutoff, _ = get_fitted(ds_key, kind, use_prot)
    df = load_df(ds_key).copy()
    df["decision"] = (fitted["prob"] >= cutoff).astype(int)

    overall_rate = float(df["decision"].mean())
    attrs_meta = GROUP_COLS.get(ds_key, [])
    attributes = []
    all_groups = []

    for col, label in attrs_meta:
        if col not in df.columns:
            continue
        if selected_attr != "all" and col != selected_attr:
            continue
        tbl = compare_groups(df, col, "decision", "approved")
        fl_ratio = fairlearn_check(df, col, "decision", "approved")
        rows = []
        for _, r in tbl.iterrows():
            ratio = _clean_val(r["ratio_vs_best"])
            n_val = int(r["n"])
            if not r["testable"]:
                tier = "tiny"
            elif ratio is None:
                tier = "fine"
            elif ratio < SCREEN_RATIO:
                tier = "flag"
            elif ratio < PRACTICAL_RATIO:
                tier = "review"
            else:
                tier = "fine"

            row_obj = {
                "attribute_key": col,
                "attribute_label": label,
                "group": str(r["group"]),
                "n": n_val,
                "approved": int(r["approved"]),
                "selection_rate": round(float(r["selection_rate"]), 4),
                "selection_rate_pct": round(float(r["selection_rate"]) * 100, 1),
                "ratio_vs_best": round(float(ratio), 3) if ratio is not None else None,
                "p_value": _clean_val(r["p_value"]),
                "tpr": _clean_val(r["tpr"]),
                "tpr_gap": _clean_val(r["tpr_gap"]),
                "reference_group": _clean_val(r["reference_group"]),
                "verdict": str(r["verdict"]),
                "testable": bool(r["testable"]),
                "tiny_group": n_val < MIN_GROUP,
                "tier": tier,
            }
            rows.append(row_obj)
            all_groups.append(row_obj)

        attributes.append({
            "key": col,
            "label": label,
            "fairlearn_ratio": round(fl_ratio, 4) if fl_ratio is not None else None,
            "groups": rows,
        })

    rates = [g["selection_rate"] for g in all_groups if g["testable"]]
    ratios = [g["ratio_vs_best"] for g in all_groups if g["ratio_vs_best"] is not None]
    pvals = [g["p_value"] for g in all_groups if g["p_value"] is not None]

    log_activity(
        "Group Fairness Audit",
        f"{DATASETS[ds_key]['short']} · {MODELS[kind]} · {'Protected ON' if use_prot else 'Unaware'}",
        status="info",
    )

    return jsonify({
        "dataset": ds_key,
        "model": kind,
        "use_protected": use_prot,
        "available_attributes": [{"key": c, "label": l} for c, l in attrs_meta if c in df.columns],
        "summary": {
            "overall_rate": round(overall_rate, 4),
            "lowest_rate": round(min(rates), 4) if rates else None,
            "highest_rate": round(max(rates), 4) if rates else None,
            "min_ratio": round(min(ratios), 3) if ratios else None,
            "min_p_value": min(pvals) if pvals else None,
            "total_rows": len(df),
        },
        "attributes": attributes,
    })


@app.route("/api/legal/check", methods=["POST"])
def api_legal_check():
    data = request.get_json(force=True)
    jurisdiction = data.get("jurisdiction", "US")
    audit_date = data.get("audit_date", DEFAULT_AUDIT_DATE)
    test_type = data.get("test_type", "ratio_test")
    finding_text = data.get("finding_text", "").strip()
    run_llm = bool(data.get("run_llm", False))

    if not finding_text:
        return jsonify({"error": "finding_text is required"}), 400

    cards = retrieve_cards(jurisdiction, test_type, finding_text, top_k=3)
    checked = []
    for c in cards:
        label, reason = check_applicability(c, jurisdiction, "lending", audit_date)
        checked.append({"card": c, "label": label, "reason": reason})

    explanation = None
    verification = None
    if run_llm and checked:
        if os.getenv("GROQ_API_KEY"):
            try:
                from report_writer import write_report
                explanation = write_report(finding_text, checked)
                verification = verify_report(explanation, checked)
            except Exception as exc:
                explanation = f"[Groq API error: {exc}]"
        else:
            explanation = None

    log_activity(
        "Legal Compliance Check",
        f"{jurisdiction} ({audit_date}) · {test_type} · {len(checked)} cards matched",
        status="info",
    )

    return jsonify({
        "jurisdiction": jurisdiction,
        "audit_date": audit_date,
        "test_type": test_type,
        "cards": checked,
        "explanation": explanation,
        "verification": verification,
        "groq_available": bool(os.getenv("GROQ_API_KEY")),
    })


@app.route("/api/cards")
def api_cards():
    jurisdiction = request.args.get("jurisdiction", "ALL")
    audit_date = request.args.get("audit_date", DEFAULT_AUDIT_DATE)
    cards = load_cards()
    out = []
    for c in cards:
        if jurisdiction != "ALL" and c["jurisdiction"] != jurisdiction:
            continue
        target_jur = c["jurisdiction"] if jurisdiction == "ALL" else jurisdiction
        label, reason = check_applicability(c, target_jur, "lending", audit_date)
        out.append({"card": c, "label": label, "reason": reason})
    return jsonify({"cards": out, "total": len(out)})


@app.route("/api/scenarios")
def api_scenarios():
    sc_csv = ROOT / "reports" / "scenario_fairness.csv"
    fl_csv = ROOT / "reports" / "scenario_flips.csv"
    tr_json = ROOT / "reports" / "scenario_planted_truth.json"

    fairness_rows = []
    if sc_csv.exists():
        sdf = pd.read_csv(sc_csv)
        sdf = sdf[sdf["variant"] == "uses_protected"]
        fairness_rows = [{k: _clean_val(v) for k, v in r.items()} for r in sdf.to_dict(orient="records")]

    flip_rows = []
    if fl_csv.exists():
        fdf = pd.read_csv(fl_csv)
        fdf = fdf[fdf["variant"] == "uses_protected"]
        flip_rows = [{k: _clean_val(v) for k, v in r.items()} for r in fdf.to_dict(orient="records")]

    truth = json.loads(tr_json.read_text(encoding="utf-8")) if tr_json.exists() else {}

    return jsonify({
        "fairness": fairness_rows,
        "flips": flip_rows,
        "truth": truth,
    })


@app.route("/api/model-comparison")
def api_model_comparison():
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    prot = _compute_model_metrics(ds_key, kind, True)
    unaware = _compute_model_metrics(ds_key, kind, False)
    return jsonify({
        "dataset": ds_key,
        "model": kind,
        "uses_protected": prot,
        "unaware": unaware,
        "note": "Removing protected columns does not automatically eliminate disparity because proxy features can still encode group membership.",
    })


@app.route("/api/proxies")
def api_proxies():
    ds_key = request.args.get("dataset", "german")
    proxy_csv = ROOT / "reports" / "proxy_results.csv"
    top_csv = ROOT / "reports" / "proxy_top_columns.csv"
    if not proxy_csv.exists():
        return jsonify({"proxies": [], "note": "Run python src/04b_proxy_check.py to generate proxy findings."})

    pdf = pd.read_csv(proxy_csv)
    pdf = pdf[pdf["dataset"] == ds_key]
    tdf = pd.read_csv(top_csv) if top_csv.exists() else pd.DataFrame()

    proxies = []
    for _, r in pdf.iterrows():
        top_cols = []
        if not tdf.empty:
            sub_t = tdf[tdf["target"] == r["target"]].head(5)
            top_cols = [{"column": str(tr["column"]), "importance": round(float(tr["importance"]), 4)} for _, tr in sub_t.iterrows()]
        proxies.append({
            "target": str(r["target"]),
            "column": str(r["column"]),
            "auc": _clean_val(r["auc"]),
            "balanced_accuracy": _clean_val(r["balanced_accuracy"]),
            "lift_over_chance": _clean_val(r["lift_over_chance"]),
            "verdict": str(r["verdict"]),
            "reliability": str(r["reliability"]),
            "top_columns": top_cols,
        })
    return jsonify({"dataset": ds_key, "proxies": proxies})


@app.route("/api/explainability")
def api_explainability():
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    use_prot = request.args.get("use_protected", "true").lower() == "true"
    row_idx = int(request.args.get("row_idx", 0))

    if kind != "logistic":
        return jsonify({
            "supported": False,
            "message": f"Linear feature contributions are only available for Logistic Regression. '{MODELS[kind]}' does not expose direct feature weights without external SHAP.",
        })

    fitted, cutoff, features = get_fitted(ds_key, kind, use_prot)
    df = load_df(ds_key)
    if row_idx < 0 or row_idx >= len(df):
        return jsonify({"error": "Applicant index out of range"}), 404

    pipe = fitted["models"][fitted["fold"][row_idx]]
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feat_names = list(prep.get_feature_names_out())
    coefs = clf.coef_[0]

    X_row = df.iloc[[row_idx]][features]
    X_trans = prep.transform(X_row)[0]
    contribs = X_trans * coefs

    local_pairs = sorted(
        [{"feature": n, "contribution": round(float(c), 4)} for n, c in zip(feat_names, contribs) if abs(c) > 1e-4],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )[:10]

    global_pairs = sorted(
        [{"feature": n, "weight": round(float(w), 4)} for n, w in zip(feat_names, coefs)],
        key=lambda x: abs(x["weight"]),
        reverse=True,
    )[:10]

    return jsonify({
        "supported": True,
        "row_idx": row_idx,
        "local_contributions": local_pairs,
        "global_weights": global_pairs,
        "limitation": "Contributions reflect standardized log-odds weights from the out-of-fold Logistic Regression pipeline, not causal effects.",
    })


@app.route("/api/reports/generate/json")
def api_generate_report_json():
    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    jur = request.args.get("jurisdiction", "US")
    audit_date = request.args.get("audit_date", DEFAULT_AUDIT_DATE)
    payload = _build_audit_payload(ds_key, kind, jur, audit_date)
    buf = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    return send_file(buf, mimetype="application/json", as_attachment=True, download_name=f"rgca_audit_{ds_key}_{kind}.json")


@app.route("/api/reports/generate/pdf")
def api_generate_report_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ds_key = request.args.get("dataset", "german")
    kind = request.args.get("model", "logistic")
    jur = request.args.get("jurisdiction", "US")
    audit_date = request.args.get("audit_date", DEFAULT_AUDIT_DATE)
    payload = _build_audit_payload(ds_key, kind, jur, audit_date)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="RGCA Compliance Audit Report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("RGCA — Regulatory-Grounded Compliance Audit Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Audit ID: {payload['audit_id']} | Generated: {payload['generated_at']} | Audit Date: {audit_date}", styles["Normal"]),
        Paragraph(f"Dataset: {payload['dataset']['label']} (n={payload['dataset']['rows']}) | Model: {payload['model']['label']} | Jurisdiction: {jur}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("1. Model Performance (5-Fold Out-of-Fold)", styles["Heading2"]),
    ]

    p_prot = payload["performance"]["uses_protected"]
    p_un = payload["performance"]["unaware"]
    perf_table = Table([
        ["Variant", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["Uses Protected", p_prot["accuracy"], p_prot["precision"], p_prot["recall"], p_prot["f1"], p_prot["roc_auc"]],
        ["Unaware", p_un["accuracy"], p_un["precision"], p_un["recall"], p_un["f1"], p_un["roc_auc"]],
    ])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14243A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.extend([perf_table, Spacer(1, 14), Paragraph("2. Group Fairness Findings", styles["Heading2"])])

    fair_data = [["Attribute", "Group", "n", "Rate", "Ratio", "p-value", "Verdict"]]
    for r in payload["group_fairness"][:15]:
        fair_data.append([
            r["attribute"], r["group"], r["n"],
            f"{r['selection_rate']*100:.1f}%",
            r["ratio_vs_best"] if r["ratio_vs_best"] is not None else "—",
            f"{r['p_value']:.2e}" if r["p_value"] is not None else "—",
            r["verdict"][:28],
        ])
    fair_table = Table(fair_data)
    fair_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14243A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([fair_table, Spacer(1, 14), Paragraph("3. Legal Applicability Summary", styles["Heading2"])])

    leg_data = [["Card ID", "Clause", "Effective", "Label"]]
    for l in payload["legal_applicability"]:
        leg_data.append([l["id"], l["clause"][:35], l["effective_from"] or "Null", l["label"]])
    leg_table = Table(leg_data)
    leg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14243A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([leg_table, Spacer(1, 14), Paragraph("4. Warnings & Limitations", styles["Heading2"])])
    for w in payload["warnings_and_limitations"]:
        story.append(Paragraph(f"• {w}", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"rgca_audit_{ds_key}_{kind}.pdf")


@app.route("/api/reports/data")
def api_report_data():
    name = request.args.get("name", "fairness_results.csv")
    safe_path = (ROOT / "reports" / Path(name).name).resolve()
    if not safe_path.exists() or safe_path.parent != (ROOT / "reports").resolve():
        return jsonify({"error": "Report file not found"}), 404

    if safe_path.suffix.lower() == ".json":
        return jsonify({
            "type": "json",
            "name": safe_path.name,
            "data": json.loads(safe_path.read_text(encoding="utf-8")),
        })

    df = pd.read_csv(safe_path)
    limit = int(request.args.get("limit", 100))
    rows = [{k: _clean_val(v) for k, v in r.items()} for r in df.head(limit).to_dict(orient="records")]
    return jsonify({
        "type": "csv",
        "name": safe_path.name,
        "columns": list(df.columns),
        "rows": rows,
        "total_rows": len(df),
    })


@app.route("/api/reports/download/<path:filename>")
def api_report_download(filename):
    return send_from_directory(ROOT / "reports", Path(filename).name, as_attachment=True)


@app.route("/api/settings/clear-cache", methods=["POST"])
def api_clear_cache():
    _DF_CACHE.clear()
    _FIT_CACHE.clear()
    log_activity("System Cache Cleared", "Dataset and OOF model caches reset", status="info")
    return jsonify({"ok": True, "message": "Dataset and model caches cleared."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
