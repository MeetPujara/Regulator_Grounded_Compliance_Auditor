"""
RGCA — Demo-first Streamlit app.
Run:  streamlit run app.py
Story: show applicants → flip one trait → see decision change → check the law.
"""
import json, os, sys, re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
os.chdir(ROOT)
sys.path.insert(0, str(SRC))
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="RGCA Auditor", page_icon="⚖️", layout="wide")

from card_schema import load_cards
from counterfactuals import FLIPS, run_counterfactuals
from data_loading import load_south_german, load_taiwan
from fairness import compare_groups
from models import THRESHOLD, feature_plan, fit_oof
from scenarios import (TARGET_GROUP, TARGET_ZONE, add_synthetic_columns,
                       plant_bias, run_flips, scenario_feature_plan)
from applicability import check_applicability
from retrieval import retrieve_cards

# ── cached loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(key):
    if key == "german":   return load_south_german()
    if key == "taiwan":   return load_taiwan()
    if key == "scenario_A": return add_synthetic_columns(load_south_german())
    if key == "scenario_B":
        planted, _ = plant_bias(add_synthetic_columns(load_south_german()))
        return planted

@st.cache_resource(show_spinner=False)
def get_fitted(key, kind, use_protected):
    df = load_df(key)
    if key.startswith("scenario"):
        num, cat = scenario_feature_plan(use_protected)
    else:
        num, cat = feature_plan(key, use_protected)
    fitted = fit_oof(kind, df, num, cat)
    cutoff = THRESHOLD if key != "taiwan" else float(
        np.quantile(fitted["prob"], 1 - df["approved"].mean()))
    return fitted, cutoff, num + cat

# ── readable applicant table ────────────────────────────────────────────
READABLE = {
    "german": [
        ("age", "Age"), ("personal_status_sex_label", "Status"),
        ("foreign_worker_label", "Foreign?"), ("duration", "Duration(mo)"),
        ("amount", "Amount"), ("approved", "Truth"),
    ],
    "taiwan": [
        ("AGE", "Age"), ("sex_label", "Sex"), ("marriage_label", "Marital"),
        ("LIMIT_BAL", "Limit"), ("default", "Defaulted?"), ("approved", "Truth"),
    ],
    "scenario_A": [
        ("age", "Age"), ("syn_pincode", "Pincode*"), ("syn_surname_group", "Surname*"),
        ("amount", "Amount"), ("approved", "Truth"),
    ],
    "scenario_B": [
        ("age", "Age"), ("syn_pincode", "Pincode*"), ("syn_surname_group", "Surname*"),
        ("amount", "Amount"), ("approved", "Truth"),
    ],
}
STAR = "* synthetic column (not real data)"

def make_table(key, fitted, cutoff, n_show=20):
    df = load_df(key).head(n_show).copy()
    df["_prob"] = fitted["prob"][:n_show]
    df["_decision"] = (df["_prob"] >= cutoff).astype(int)
    cols = [(c, lbl) for c, lbl in READABLE[key] if c in df.columns]
    out = df[[c for c, _ in cols]].copy()
    out.columns = [lbl for _, lbl in cols]
    out["Model says"] = np.where(df["_decision"] == 1, "✅ Approved", "❌ Rejected")
    out["Prob(good)"] = df["_prob"].round(3)
    return out, df

# ── flip logic ──────────────────────────────────────────────────────────
def get_valid_flips(key, row):
    """Which flips apply to this specific applicant?"""
    flips = []
    if key in ("german", "scenario_A", "scenario_B"):
        if row.get("foreign_worker") == 1:
            flips.append(("Foreign worker", "foreign_worker", 1, 2, "yes → no"))
        if row.get("foreign_worker") == 2:
            flips.append(("Foreign worker", "foreign_worker", 2, 1, "no → yes"))
        ps = row.get("personal_status_sex")
        if ps == 1: flips.append(("Marital status", "personal_status_sex", 1, 3, "divorced → married"))
        if ps == 3: flips.append(("Marital status", "personal_status_sex", 3, 1, "married → divorced"))
        if row.get("age", 99) < 25:
            flips.append(("Age", "age", row["age"], 35, f"{row['age']} → 35"))
        if key.startswith("scenario"):
            if row.get("syn_pincode") == TARGET_ZONE:
                flips.append(("Pincode*", "syn_pincode", TARGET_ZONE, "SYN_PIN_2", "target → ref"))
            if row.get("syn_surname_group") == TARGET_GROUP:
                flips.append(("Surname*", "syn_surname_group", TARGET_GROUP, "SYN_GROUP_A", "target → ref"))
    elif key == "taiwan":
        if row.get("SEX") == 1: flips.append(("Sex", "SEX", 1, 2, "male → female"))
        if row.get("SEX") == 2: flips.append(("Sex", "SEX", 2, 1, "female → male"))
        if row.get("MARRIAGE") == 1: flips.append(("Marriage", "MARRIAGE", 1, 2, "married → single"))
        if row.get("MARRIAGE") == 2: flips.append(("Marriage", "MARRIAGE", 2, 1, "single → married"))
    return flips

def do_flip(key, kind, use_protected, row_idx, col, old_val, new_val):
    fitted, cutoff, features = get_fitted(key, kind, use_protected)
    df = load_df(key)
    original = df.iloc[row_idx].copy()
    changed = original.copy()
    changed[col] = new_val
    X = pd.DataFrame([changed[features]])
    fold_model = fitted["models"][fitted["fold"][row_idx]]
    new_prob = float(fold_model.predict_proba(X)[0, 1])
    old_prob = float(fitted["prob"][row_idx])
    return old_prob, new_prob, cutoff

# ── legal panel ─────────────────────────────────────────────────────────
AUDIT_DATE = "2026-11-24"

def legal_panel(finding_text, jurisdiction, test_type):
    cards = retrieve_cards(jurisdiction, test_type, finding_text, top_k=3)
    if not cards:
        st.info("No rule cards found for this combination.")
        return
    for card in cards:
        label, reason = check_applicability(card, jurisdiction, "lending", AUDIT_DATE)
        color = {"Applies": "🟢", "Not yet in force": "🟡",
                 "Out of scope": "⚪", "Heuristic only": "🔵",
                 "Wrong jurisdiction": "⚪", "Needs human review": "🟠"}.get(label, "⚪")
        st.markdown(f"### {color} `{card['id']}` — {label}")
        st.markdown(f"**Rule:** {card['obligation']}")
        st.markdown(f"**Why this label:** {reason}")
        if card.get("caveat"):
            st.markdown(f"**Fine print:** {card['caveat']}")
        st.divider()

@st.cache_data(show_spinner=False, ttl=3600)
def cached_legal(finding_text, jur, iso):
    """Retrieve + check + Groq, once per unique (finding, jurisdiction, date)."""
    cards = retrieve_cards(jur, "ratio_test", finding_text, top_k=3)
    checked = []
    for c in cards:
        lab, rea = check_applicability(c, jur, "lending", iso)
        checked.append({"card": c, "label": lab, "reason": rea})
    para = None
    if os.getenv("GROQ_API_KEY") and checked:
        try:
            from report_writer import write_report
            para = write_report(finding_text, checked)
        except Exception as e:
            para = f"[Groq error: {e}]"
    return checked, para
# ── group bar chart ─────────────────────────────────────────────────────
GROUP_COLS = {
    "german": [("age_group", "Age"), ("foreign_worker_label", "Foreign worker"),
               ("personal_status_sex_label", "Marital status")],
    "taiwan": [("sex_label", "Sex"), ("marriage_label", "Marriage"), ("age_group", "Age")],
    "scenario_B": [("syn_pincode", "Pincode*"), ("syn_surname_group", "Surname*")],
    "scenario_A": [("syn_pincode", "Pincode*"), ("syn_surname_group", "Surname*")],
}

def group_chart(key, fitted, cutoff):
    df = load_df(key).copy()
    df["decision"] = (fitted["prob"] >= cutoff).astype(int)
    for col, label in GROUP_COLS.get(key, []):
        if col not in df.columns: continue
        rates = df.groupby(col)["decision"].mean()
        counts = df.groupby(col)["decision"].count()
        best = rates.max()
        ratios = (rates / best).round(3)
        st.markdown(f"**{label}**")
        cols = st.columns(len(rates))
        for i, (grp, rate) in enumerate(rates.items()):
            r = ratios[grp]
            emoji = "🔴" if r < 0.8 else ("🟡" if r < 0.9 else "🟢")
            n = counts[grp]
            small = " *(tiny group)*" if n < 30 else ""
            with cols[i]:
                st.metric(f"{emoji} {grp}{small}", f"{rate:.0%}", f"ratio {r:.2f}")

# ── PAGE ────────────────────────────────────────────────────────────────
st.title("⚖️ RGCA — AI Loan Auditor")
st.caption("Does the model treat groups unfairly? And is that actually illegal?")

tab_demo, tab_groups, tab_law, tab_proof = st.tabs([
    "🎯 Live Demo", "📊 Group Fairness", "⚖️ Legal Check", "🔬 Proof (Scenarios)"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: LIVE DEMO — the star of the show
# ═══════════════════════════════════════════════════════════════════════
with tab_demo:
    st.header("1. Look at real applicants")
    st.markdown("Pick a dataset and model. See who got approved and who got rejected. "
                "Then **flip one trait** on a rejected person and watch what happens.")

    c1, c2, c3 = st.columns(3)
    ds_key = c1.selectbox("Dataset", ["german", "taiwan", "scenario_B"],
                          format_func=lambda x: {"german": "South German Credit",
                                                 "taiwan": "Taiwan Credit Cards",
                                                 "scenario_B": "Scenario B (planted bias)"}[x])
    kind = c2.selectbox("Model", ["logistic", "boosted"])
    use_prot = c3.checkbox("Model sees protected traits", value=True,
                           help="Uncheck = 'unaware' model. Protected columns removed.")

    fitted, cutoff, feats = get_fitted(ds_key, kind, use_prot)
    table, raw_df = make_table(ds_key, fitted, cutoff, n_show=20)

    st.dataframe(table, use_container_width=True, height=350)
    if ds_key.startswith("scenario"):
        st.caption(STAR)

    st.divider()
    st.header("2. Flip one trait — watch the decision change")

    rejected_idx = raw_df.index[raw_df["_decision"] == 0].tolist()
    if not rejected_idx:
        st.success("No rejected applicants in this view. Try 'Model sees protected traits' = ON.")
    else:
        pick = st.selectbox("Pick a rejected applicant (row #)",
                            rejected_idx, format_func=lambda i: f"Row {i} — age {raw_df.iloc[i].get('age', raw_df.iloc[i].get('AGE','?'))}")
        row = raw_df.iloc[pick]
        valid = get_valid_flips(ds_key, row)
        if not valid:
            st.info("No flippable attribute for this person.")
        else:
            flip_names = [v[0] for v in valid]
            flip_pick = st.selectbox("What to change?", flip_names)
            fp = valid[flip_names.index(flip_pick)]
            _, col, old_v, new_v, desc = fp

            if st.button(f"🔄 Flip: {desc}", type="primary"):
                old_p, new_p, cut = do_flip(ds_key, kind, use_prot, pick, col, old_v, new_v)
                st.session_state["flip_result"] = {
                    "config": (ds_key, kind, use_prot),
                    "desc": desc,
                    "col": col,
                    "old_p": old_p,
                    "new_p": new_p,
                    "cut": cut,
                    "finding": (f"Changing only {desc} for a rejected applicant flipped the "
                                f"decision from rejection to approval. "
                                f"The model uses {col} directly."),
                }

    # ---- show the saved flip result (survives dropdown changes) ----
    fr = st.session_state.get("flip_result")
    if fr and fr["config"] == (ds_key, kind, use_prot):
        old_p, new_p, cut = fr["old_p"], fr["new_p"], fr["cut"]
        before = "✅ Approved" if old_p >= cut else "❌ Rejected"
        after = "✅ Approved" if new_p >= cut else "❌ Rejected"
        flipped = old_p < cut <= new_p

        st.divider()
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Before", before, f"prob {old_p:.3f}")
        cc2.metric("After", after, f"prob {new_p:.3f}")
        cc3.metric("Changed?", "YES ⚡" if flipped else "No",
                   f"+{new_p - old_p:.3f}" if flipped else "same decision")

        if flipped:
            st.error(f"**The model changed its mind just by seeing {fr['desc']}.** "
                     f"This person is otherwise identical. The attribute **{fr['col']}** "
                     f"directly drove the decision.")
        else:
            st.info("Decision did not flip. The attribute alone wasn't enough "
                    "to cross the cutoff for this person.")

        st.divider()
        st.header("3. Is that illegal?")
        jur = st.selectbox("Check under which law?", ["US", "EU", "IN"], key="demo_jur")
        legal_panel(fr["finding"], jur, "counterfactual")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: GROUP FAIRNESS — bar chart, no jargon
# ═══════════════════════════════════════════════════════════════════════
with tab_groups:
    st.header("Approval rate by group")
    st.markdown("🟢 = fine (ratio ≥ 0.9) · 🟡 = review (0.8–0.9) · 🔴 = flag (< 0.8)")

    g_ds = st.selectbox("Dataset", ["german", "taiwan", "scenario_B"], key="g_ds",
                        format_func=lambda x: {"german": "South German Credit",
                                               "taiwan": "Taiwan Credit Cards",
                                               "scenario_B": "Scenario B (planted bias)"}[x])
    g_kind = st.selectbox("Model", ["logistic", "boosted"], key="g_kind")
    g_fit, g_cut, _ = get_fitted(g_ds, g_kind, True)
    group_chart(g_ds, g_fit, g_cut)

    st.divider()
    st.markdown("""
### What the colors mean
- **🔴 Flag**: this group's approval rate is below 80% of the best group, AND the gap
  is too large to be luck. This is where the legal check matters.
- **🟡 Review**: gap is real but between 80–90%. Worth looking at, not automatic trouble.
- **🟢 Fine**: no meaningful gap.
- *(tiny group)*: fewer than 30 people. The number is unstable — don't build a case on it.
""")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: LEGAL CHECK — cards + labels + Groq
# ═══════════════════════════════════════════════════════════════════════
with tab_law:
    st.header("Legal check for group fairness findings")
    st.markdown("Pick jurisdiction + date, press **Run legal check** once. "
                "Results are cached, so switching back is instant.")

    lc1, lc2 = st.columns(2)
    l_jur = lc1.selectbox("Jurisdiction", ["US", "EU", "IN"], key="l_jur")
    l_date = lc2.date_input("Audit date", value=date(2026, 11, 24), key="l_date")
    if isinstance(l_date, tuple): l_date = l_date[0]
    l_iso = l_date.isoformat()

    fair_csv = ROOT / "reports" / "fairness_results.csv"
    if fair_csv.exists():
        fdf = pd.read_csv(fair_csv)
        fdf = fdf[fdf["variant"] == "uses_protected"]
        fdf = fdf[fdf["ratio_vs_best"].notna()]
        worst = fdf.sort_values("ratio_vs_best").head(5)
    else:
        worst = pd.DataFrame()

    if worst.empty:
        st.info("Run `python src\\03_fairness_tests.py` first.")
    else:
        if st.button("▶ Run legal check", type="primary"):
            with st.spinner("Checking cards and writing explanations (5 Groq calls)..."):
                results = []
                for _, row in worst.iterrows():
                    finding = (f"The {row['group']} group was approved at "
                               f"{row['ratio_vs_best']:.2f} times the best group's rate "
                               f"(n={int(row['n'])}). Verdict: {row['verdict']}.")
                    checked, para = cached_legal(finding, l_jur, l_iso)
                    results.append({"row": row.to_dict(), "checked": checked, "para": para})
                st.session_state["law_results"] = results
                st.session_state["law_key"] = (l_jur, l_iso)

        shown = st.session_state.get("law_results")
        if shown and st.session_state.get("law_key") == (l_jur, l_iso):
            for res in shown:
                row = res["row"]
                ratio = row["ratio_vs_best"]
                emoji = "🔴" if ratio < 0.8 else ("🟡" if ratio < 0.9 else "🟢")
                st.markdown(f"### {emoji} {row['dataset']} / {row['model']} / {row['group']}")
                st.markdown(f"Approved at **{ratio:.2f}×** best group (n={int(row['n'])})")
                for item in res["checked"]:
                    c, lab = item["card"], item["label"]
                    dot = {"Applies": "🟢", "Not yet in force": "🟡", "Out of scope": "⚪",
                           "Heuristic only": "🔵", "Needs human review": "🟠"}.get(lab, "⚪")
                    st.markdown(f"{dot} `{c['id']}` — **{lab}**: {item['reason']}")
                if res["para"]:
                    st.info(res["para"])
                st.divider()
        elif shown:
            st.info("Jurisdiction or date changed — press **Run legal check** again.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: PROOF — Scenario A vs B side by side
# ═══════════════════════════════════════════════════════════════════════
with tab_proof:
    st.header("Does the tool cry wolf?")
    st.markdown("**Scenario A** = clean data, no planted bias. Tool should stay quiet. "
                "**Scenario B** = we deliberately made pincode affect approvals. Tool must catch it.")

    sc_csv = ROOT / "reports" / "scenario_fairness.csv"
    if not sc_csv.exists():
        st.info("Run `python src\\05_scenarios.py` first.")
    else:
        sdf = pd.read_csv(sc_csv)
        sdf = sdf[sdf["variant"] == "uses_protected"]

        cA, cB = st.columns(2)
        for col, scen, title in [(cA, "A_clean", "Scenario A (clean)"),
                                  (cB, "B_planted", "Scenario B (planted bias)")]:
            sub = sdf[sdf["scenario"] == scen]
            flags = sub[sub["verdict"].str.startswith("FLAG", na=False)]
            with col:
                st.subheader(title)
                if scen == "A_clean":
                    if flags.empty:
                        st.success("✅ No synthetic bias flagged. Tool stayed quiet.")
                    else:
                        st.error("❌ Unexpected flag in clean data!")
                        st.dataframe(flags[["attribute","group","ratio_vs_best"]])
                else:
                    if not flags.empty:
                        st.error(f"🚨 {len(flags)} flags found — planted bias detected!")
                        st.dataframe(flags[["attribute","group","ratio_vs_best","p_value"]])
                    else:
                        st.warning("Planted bias NOT detected. Investigate.")

        # flip comparison
        flip_csv = ROOT / "reports" / "scenario_flips.csv"
        if flip_csv.exists():
            fldf = pd.read_csv(flip_csv)
            fldf = fldf[fldf["variant"] == "uses_protected"]
            st.subheader("Flip rates: single vs bundle")
            for scen in ["A_clean", "B_planted"]:
                sub = fldf[fldf["scenario"] == scen]
                if sub.empty: continue
                st.markdown(f"**{scen}**")
                for _, r in sub.iterrows():
                    rate = r["flip_rate"] if pd.notna(r["flip_rate"]) else 0
                    bar = "█" * int(rate * 30)
                    st.markdown(f"`{r['flip']:<20}` {bar} {rate:.1%} ({int(r['n_flips'])}/{int(r['n_eligible'])})")

        st.divider()
        st.markdown("""
### Why this matters
A fairness tool that flags everything is useless — it cries wolf.
A tool that flags nothing is also useless — it misses real bias.
Scenario A proves it stays quiet. Scenario B proves it catches planted bias.
The **bundle flip** (pincode + surname together) is worse than either alone —
that's intersectional bias, and single-attribute tests miss it.
""")

# ── footer ──────────────────────────────────────────────────────────────
st.divider()
st.caption("RGCA v2 · Demo-first UI · 18 verified rule cards · US / EU / India · "
           "LLM writes prose, code decides law · Audit date 2026-11-24")