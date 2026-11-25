"""
Step 6: write the rule cards, then check them.
Run from the project root:   python src\06_write_cards.py

Writes: cards/schema.json and cards/<ID>.json (18 cards)
Prints: a validation report + which source file each card resolved to.

Every obligation below comes from facts verified in the saved source files.
Where we only verified a date (the Reg B amendment), the card says only that.
"""
import json
from collections import Counter
from pathlib import Path

from card_schema import SCHEMA, load_cards, validate

AUDIT_DATE = "2026-11-24"
RETRIEVED = "2026-11-24"


def C(id, jurisdiction, clause, source_url, source_file, status, applies_to,
      obligation, test_type, effective_from=None, caveat="", threshold=None,
      to_verify=(), uncertain=False):
    """Build one card with the standard boxes."""
    return {"id": id, "jurisdiction": jurisdiction, "clause": clause,
            "source_url": source_url, "source_file": source_file, "status": status,
            "applies_to": applies_to, "effective_from": effective_from,
            "obligation": obligation, "caveat": caveat, "test_type": test_type,
            "threshold": threshold, "to_verify": list(to_verify),
            "applicability_uncertain": uncertain, "retrieved_on": RETRIEVED}


CARDS = [
    # ---------------------------- US (7) -----------------------------------
    C("US-EEOC-FOURFIFTHS", "US", "29 CFR 1607.4(D)",
      "https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607/section-1607.4",
      ["*1607*", "*Information*on*impact*"], "heuristic", ["employment"],
      "A selection rate for any group below four-fifths (80%) of the highest "
      "group's rate is generally regarded as evidence of adverse impact.",
      ["ratio_test"], threshold=0.8,
      caveat="Employment rule (UGESP), not lending law; used here only as a "
             "screening heuristic. Not a bright line: smaller gaps can count if "
             "statistically and practically significant, larger gaps may not if "
             "the numbers are small.",
      to_verify=["effective date (published 1978-08-25, amended 1981-12-31)"]),

    C("US-ECOA-PROHIBITED", "US", "15 U.S.C. 1691(a), (b)(3)",
      "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section1691",
      ["*1691*", "*ECOA*", "*Scope*of*prohibition*"], "binding_law", ["lending"],
      "A creditor may not discriminate against an applicant on the basis of race, "
      "color, religion, national origin, sex, marital status, age, or receipt of "
      "public assistance income.",
      ["counterfactual", "ratio_test"],
      caveat="Age may be used in an empirically derived, demonstrably and "
             "statistically sound scoring system, but an elderly applicant's age "
             "may never be given a negative value. Held copy is Cornell LII, not "
             "a .gov site.",
      to_verify=["effective date (enacted 1974-10-28)"]),

    C("US-ECOA-ADVERSE", "US", "15 U.S.C. 1691(d)(2)-(3)",
      "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section1691",
      ["*1691*", "*ECOA*", "*Scope*of*prohibition*"], "binding_law", ["lending"],
      "An applicant who is denied credit is entitled to a statement of reasons, "
      "which satisfies the law only if it states the specific reasons.",
      ["explanation_faithfulness"],
      caveat="Statutory basis for the detailed rule in Regulation B 12 CFR 1002.9.",
      to_verify=["effective date (enacted 1974-10-28)"]),

    C("US-REGB-ADVERSE", "US",
      "12 CFR 1002.9(a)(2), (b)(2); Official Interpretations 9(b)(2)-1 to -4",
      "https://www.consumerfinance.gov/rules-policy/regulations/1002/9/",
      ["*1002.9*", "*Notifications*"], "regulation", ["lending"],
      "Adverse-action reasons must be specific, state the principal reasons, and "
      "accurately describe the factors actually considered or scored.",
      ["explanation_faithfulness"],
      caveat="Generic reasons such as 'did not meet internal standards' are "
             "insufficient. FCRA credit-score key factors do not replace these "
             "reasons. More than four reasons is unlikely to help. Notification "
             "is due within 30 days of a completed application.",
      to_verify=["effective date not stated on the page",
                 "whether the 2026 disparate-impact amendment changed this section"]),

    C("US-CFPB-CIRCULAR-2022-03", "US", "CFPB Circular 2022-03 (26 May 2022)",
      "https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-complex-algorithms/",
      ["*2022-03*", "*Circular*2022*", "*circular*"], "guidance", ["lending"],
      "Using a complex algorithm does not excuse a creditor from giving specific, "
      "accurate reasons for an adverse action.",
      ["explanation_faithfulness"], effective_from="2022-05-26",
      caveat="Regulator guidance, not a statute."),

    C("US-REGB-DISPARATE-IMPACT", "US",
      "Regulation B final rule (disparate impact), Federal Register",
      "https://www.federalregister.gov/",
      ["*Federal*Register*", "*disparate*", "*Reg*B*final*"],
      "regulation", ["lending"],
      "This final rule amends Regulation B on disparate impact and takes effect "
      "on 21 July 2026.",
      ["ratio_test", "governance"], effective_from="2026-07-21",
      caveat="Only the effective date was transcribed from the source. Do not "
             "assert specific obligations from this card without reading the rule.",
      to_verify=["exact Federal Register citation and URL",
                 "detailed scope of the amendment"]),

    C("US-NYC-LL144", "US", "NYC Local Law 144 of 2021 (bias audit); section 2",
      "https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page",
      ["*144*", "*AEDT*", "*Local*Law*"], "binding_law", ["employment"],
      "Employers using automated employment decision tools must have a bias audit "
      "performed before use.",
      ["ratio_test", "governance"], effective_from="2023-01-01",
      caveat="Hiring law, not lending: our deliberate out-of-scope example. The "
             "law text says 'disparate impact'; the impact-ratio detail is in the "
             "city's separate rules, which we do not hold. Enforcement began "
             "5 July 2023.",
      to_verify=["DCWP final rules text (impact-ratio definition) not held"]),

    # ---------------------------- EU (4) -----------------------------------
    C("EU-AIA-ANNEX3-5B", "EU",
      "Regulation (EU) 2024/1689, Annex III, point 5(b)",
      "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
      ["*2024_1689*", "*2024*1689*"], "binding_law", ["lending"],
      "AI systems used to evaluate the creditworthiness of natural persons or to "
      "establish their credit score are classified as high-risk, except when used "
      "to detect financial fraud.",
      ["high_risk_classification", "governance"], effective_from="2027-12-02",
      caveat="Application date set by the Digital Omnibus, Regulation (EU) "
             "2026/1744; the original 2024 text shows an earlier date. Fraud "
             "detection is excluded."),

    C("EU-AIA-ART10", "EU",
      "Regulation (EU) 2024/1689, Article 10(2)(f)-(g)",
      "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
      ["*2024_1689*", "*2024*1689*", "*Article*10*"], "binding_law", ["lending"],
      "Providers of high-risk AI systems must examine training, validation and "
      "testing data for possible biases and address data gaps or shortcomings.",
      ["ratio_test", "proxy_test", "governance"], effective_from="2027-12-02",
      caveat="Duty on providers of high-risk systems. For Annex III credit-scoring "
             "systems the obligations apply from 2 December 2027."),

    C("EU-AIA-ART13", "EU", "Regulation (EU) 2024/1689, Article 13(1)",
      "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
      ["*2024_1689*", "*2024*1689*", "*Article*13*"], "binding_law", ["lending"],
      "High-risk AI systems must be sufficiently transparent for deployers to "
      "interpret the system's output and use it appropriately.",
      ["transparency", "explanation_faithfulness"], effective_from="2027-12-02",
      caveat="Duty on providers, owed to deployers; not a direct explanation duty "
             "to the applicant."),

    C("EU-AIA-ART4A", "EU",
      "Article 4a, inserted by Regulation (EU) 2026/1744 (Digital Omnibus on AI)",
      "https://eur-lex.europa.eu/eli/reg/2026/1744/oj",
      ["*2026_1744*", "*2026*1744*"], "binding_law", ["lending", "employment"],
      "Providers of high-risk AI systems may exceptionally process special "
      "categories of personal data to detect and correct bias, under strict "
      "conditions including that the purpose cannot be met with synthetic or "
      "anonymised data.",
      ["governance", "data_quality"], effective_from="2026-07-27", uncertain=True,
      caveat="The Omnibus entered into force on 27 July 2026, but the application "
             "date of Article 4a itself is not confirmed. Our India-style columns "
             "are synthetic; this card does not authorise collecting real "
             "sensitive data.",
      to_verify=["application date of Article 4a within the Omnibus"]),

    # --------------------------- India (6) ---------------------------------
    C("IN-FREEAI-SUTRA4", "IN",
      "RBI FREE-AI Committee report (August 2025), Sutra 4 - Fairness and Equity",
      "https://www.rbi.org.in/",
      ["*FREE*AI*", "*FREEAI*"], "guidance", ["lending"],
      "Regulated entities should ensure their AI systems treat customers fairly "
      "and equitably.",
      ["ratio_test", "counterfactual", "governance"],
      caveat="Committee report, not binding law. Addressed to regulated entities "
             "with medium-term timelines.",
      to_verify=["exact RBI URL for the FREE-AI report",
                 "whether RBI has made any FREE-AI recommendation binding"]),

    C("IN-FREEAI-SUTRA6", "IN",
      "RBI FREE-AI Committee report (August 2025), Sutra 6 - Understandable by Design",
      "https://www.rbi.org.in/",
      ["*FREE*AI*", "*FREEAI*"], "guidance", ["lending"],
      "AI systems used by regulated entities should be understandable by design "
      "so that customers can make sense of decisions affecting them.",
      ["transparency", "explanation_faithfulness"],
      caveat="Guidance, not binding law.",
      to_verify=["exact RBI URL for the FREE-AI report"]),

    C("IN-FREEAI-REC24", "IN",
      "RBI FREE-AI Committee report (August 2025), Recommendation 24 - AI Audit Framework",
      "https://www.rbi.org.in/",
      ["*FREE*AI*", "*FREEAI*"], "guidance", ["lending", "governance"],
      "Regulated entities should put in place an AI audit framework covering data "
      "inputs, the model, and decision outputs.",
      ["governance", "ratio_test", "proxy_test"],
      caveat="Recommendation with a medium-term timeline; guidance, not a binding "
             "direction. Closest match to what this tool does.",
      to_verify=["exact RBI URL for the FREE-AI report"]),

    C("IN-FREEAI-REC18", "IN",
      "RBI FREE-AI Committee report (August 2025), Recommendation 18 - Consumer Protection",
      "https://www.rbi.org.in/",
      ["*FREE*AI*", "*FREEAI*"], "guidance", ["lending"],
      "Regulated entities should strengthen consumer protection through "
      "transparency and fairness in AI-driven decisions.",
      ["transparency", "explanation_faithfulness"],
      caveat="Guidance, not binding law.",
      to_verify=["exact RBI URL for the FREE-AI report"]),

        C("IN-RBI-DLD-PARA7", "IN",
      "RBI Digital Lending Directions 2025 (8 May 2025), paragraph 7",
      "https://www.rbi.org.in/",
      ["*Notifications*Reserve*Bank*"],
      "regulation", ["lending", "record_keeping"],
      "Lenders must collect and retain the borrower's economic profile, including "
      "at minimum age, occupation and income, and must not increase credit limits "
      "automatically.",
      ["record_keeping"], effective_from="2025-05-08",
      caveat="No fairness or bias clause: record-keeping and conduct only. "
             "Paragraphs 6 and 17 had a delayed start; the rest applied at once.",
      to_verify=["exact RBI URL for the Directions"]),

    C("IN-DPDP-ACT-S8-3", "IN",
      "Digital Personal Data Protection Act 2023 (Act 22 of 2023), section 8(3)",
      "https://www.meity.gov.in/data-protection-framework",
      ["*2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf*"],
      "binding_law", ["lending", "data_quality"],
      "A data fiduciary must ensure that personal data used to make a decision "
      "affecting a data principal is complete, accurate and consistent.",
      ["data_quality"], uncertain=True,
      caveat="The Act has no anti-discrimination clause. Section 1(2) leaves "
             "commencement to a government notification, which we do not hold. "
             "One saved file named DPDP_Act.pdf is actually the Rules 2025.",
      to_verify=["commencement notification under section 1(2)",
                 "confirm which held file is the Act and which is the Rules"]),

    C("IN-DPDP-RULES-13-3", "IN",
      "DPDP Rules 2025 (G.S.R. 846(E), 13 November 2025), Rule 13(3)",
      "https://www.meity.gov.in/data-protection-framework",
      ["*DPDP*Rule*", "*846*", "*DPDP*"],
      "regulation", ["lending", "governance", "data_quality"],
      "A Significant Data Fiduciary must verify that the algorithmic software it "
      "uses for processing personal data is unlikely to pose a risk to the rights "
      "of data principals.",
      ["governance", "ratio_test", "proxy_test"],
      effective_from="2027-05-13", uncertain=True,
      caveat="Applies only to entities notified as Significant Data Fiduciaries. "
             "Start date is derived (18 months after Gazette publication), not "
             "stated as a calendar date. Rules 1, 2 and 17-21 started 13 Nov 2025; "
             "Rule 4 starts 13 Nov 2026.",
      to_verify=["whether the lender is notified as a Significant Data Fiduciary",
                 "exact commencement date derived from the Gazette"]),
]


def main():
    cards_dir = Path("cards")
    cards_dir.mkdir(exist_ok=True)
    (cards_dir / "schema.json").write_text(
        json.dumps(SCHEMA, indent=2), encoding="utf-8")
    for card in CARDS:
        (cards_dir / f"{card['id']}.json").write_text(
            json.dumps(card, indent=2), encoding="utf-8")
    print(f"Wrote {len(CARDS)} cards + schema.json to cards/\n")

    sources = sorted(p.name for p in Path("sources").rglob("*") if p.is_file())
    print(f"Files found in sources/: {len(sources)}")
    for name in sources:
        print(f"   {name}")

    print("\nValidation:")
    seen, bad = set(), 0
    for card in load_cards():
        if card["id"] in seen:
            print(f"   {card['id']:<28} DUPLICATE ID")
            bad += 1
        seen.add(card["id"])
        errors, matched = validate(card)
        if errors:
            bad += 1
            print(f"   {card['id']:<28} ERROR")
            for e in errors:
                print(f"        - {e}")
        else:
            print(f"   {card['id']:<28} OK   -> {matched[0].name}")

    print(f"\n{len(CARDS) - bad} of {len(CARDS)} cards passed.")
    print("Status mix:", dict(Counter(c["status"] for c in CARDS)))
    print("Jurisdictions:", dict(Counter(c["jurisdiction"] for c in CARDS)))

    later = [c["id"] for c in CARDS
             if c["effective_from"] and c["effective_from"] > AUDIT_DATE]
    unsure = [c["id"] for c in CARDS if c["applicability_uncertain"]]
    print(f"\nPreview at audit date {AUDIT_DATE}:")
    print(f"   not yet in force ({len(later)}): {', '.join(later)}")
    print(f"   needs human review ({len(unsure)}): {', '.join(unsure)}")


if __name__ == "__main__":
    main()