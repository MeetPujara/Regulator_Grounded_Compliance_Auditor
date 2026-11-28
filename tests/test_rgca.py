"""
Automated test suite for RGCA – Regulatory-Grounded Compliance Auditor.
Run: python -m unittest tests/test_rgca.py -v
"""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from applicability import check_applicability
from card_schema import load_cards
from fairness import compare_groups
from verifier import verify_report
from web_app import app, get_fitted, load_df


class TestRGCADataAndModels(unittest.TestCase):
    def test_south_german_encoding_and_shape(self):
        df = load_df("german")
        self.assertEqual(len(df), 1000)
        self.assertNotIn("sex", df.columns)
        self.assertIn("personal_status_sex", df.columns)
        self.assertIn("ambiguous", str(df.loc[df["personal_status_sex"] == 2, "personal_status_sex_label"].iloc[0]).lower())

    def test_model_oof_probabilities(self):
        fitted, cutoff, _ = get_fitted("german", "logistic", True)
        probs = fitted["prob"]
        self.assertEqual(len(probs), 1000)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        self.assertEqual(cutoff, 0.5)


class TestRGCAFairnessAndLegal(unittest.TestCase):
    def test_fairness_zero_denominator_safe(self):
        df = pd.DataFrame({
            "grp": ["A"] * 35 + ["B"] * 35,
            "decision": [0] * 35 + [0] * 35,
            "approved": [1] * 35 + [1] * 35,
        })
        tbl = compare_groups(df, "grp", "decision", "approved")
        b_row = tbl[tbl["group"] == "B"].iloc[0]
        self.assertTrue(pd.isna(b_row["ratio_vs_best"]))

    def test_legal_applicability_checks(self):
        cards = {c["id"]: c for c in load_cards()}
        eu_card = cards["EU-AIA-ART10"]
        lab1, _ = check_applicability(eu_card, "US", "lending", "2026-11-24")
        self.assertEqual(lab1, "Wrong jurisdiction")
        lab2, _ = check_applicability(eu_card, "EU", "lending", "2026-11-24")
        self.assertEqual(lab2, "Not yet in force")
        lab3, _ = check_applicability(eu_card, "EU", "lending", "2028-01-01")
        self.assertEqual(lab3, "Applies")

    def test_verifier_rejects_fake_citation(self):
        res = verify_report("According to [US-FAKE-999], this is illegal.", [])
        self.assertFalse(res["overall_pass"])
        self.assertFalse(res["citation_check"][0]["exists"])


class TestRGCAFlaskRoutesAndCounterfactuals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_all_pages_return_200(self):
        for route in ["/", "/live-audit", "/group-fairness", "/counterfactuals", "/legal", "/scenarios", "/reports", "/settings"]:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed")

    def test_counterfactual_single_feature_and_invalid_value(self):
        detail = self.client.get("/api/applicant/0?dataset=german&model=logistic").get_json()
        flip = detail["flips"][0]
        target_val = flip["options"][0]["value"]

        resp = self.client.post("/api/counterfactual", json={
            "dataset": "german",
            "model": "logistic",
            "use_protected": True,
            "row_idx": 0,
            "col": flip["col"],
            "new_val": target_val,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("old_prob", data)
        self.assertIn("new_prob", data)
        self.assertIn("decision_changed", data)

        # Invalid value must be rejected with 400
        bad = self.client.post("/api/counterfactual", json={
            "dataset": "german",
            "model": "logistic",
            "use_protected": True,
            "row_idx": 0,
            "col": flip["col"],
            "new_val": 999,
        })
        self.assertEqual(bad.status_code, 400)

    def test_unaware_model_counterfactual_no_change(self):
        detail = self.client.get("/api/applicant/0?dataset=german&model=logistic&use_protected=false").get_json()
        flip = detail["flips"][0]
        target_val = flip["options"][0]["value"]

        resp = self.client.post("/api/counterfactual", json={
            "dataset": "german",
            "model": "logistic",
            "use_protected": False,
            "row_idx": 0,
            "col": flip["col"],
            "new_val": target_val,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["prob_diff"], 0.0)
        self.assertFalse(data["decision_changed"])

    def test_report_generation_json_and_pdf(self):
        j_resp = self.client.get("/api/reports/generate/json?dataset=german&model=logistic&jurisdiction=US")
        self.assertEqual(j_resp.status_code, 200)
        payload = json.loads(j_resp.data.decode("utf-8"))
        self.assertEqual(payload["dataset"]["key"], "german")

        p_resp = self.client.get("/api/reports/generate/pdf?dataset=german&model=logistic&jurisdiction=US")
        self.assertEqual(p_resp.status_code, 200)
        self.assertTrue(p_resp.data.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
