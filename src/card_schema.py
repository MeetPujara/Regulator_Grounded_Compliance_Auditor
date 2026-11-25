"""
Rule-card schema and checks (Step 6).

A card is one small JSON note about one legal rule. The boxes let code CHECK the
card instead of trusting a chatbot: right country? right topic? active on the
audit date? This file defines the boxes and the checks. Later steps import it.

No extra packages needed.
"""
import json
import re
from datetime import date
from pathlib import Path

JURISDICTIONS = ["US", "EU", "IN"]
STATUSES = ["binding_law", "regulation", "guidance", "heuristic"]
TOPICS = ["lending", "employment", "data_quality", "governance", "record_keeping"]
TEST_TYPES = ["ratio_test", "counterfactual", "proxy_test",
              "explanation_faithfulness", "transparency", "governance",
              "record_keeping", "data_quality", "high_risk_classification"]

REQUIRED = ["id", "jurisdiction", "clause", "source_url", "source_file", "status",
            "applies_to", "effective_from", "obligation", "caveat", "test_type",
            "threshold", "to_verify", "applicability_uncertain", "retrieved_on"]

ID_RE = re.compile(r"^[A-Z]{2}-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA = {
    "id": "string, unique, e.g. US-ECOA-PROHIBITED",
    "jurisdiction": JURISDICTIONS,
    "clause": "string, exact place in the law",
    "source_url": "string, official link (http...)",
    "source_file": "list of glob patterns, resolved inside sources/",
    "status": STATUSES,
    "applies_to": TOPICS,
    "effective_from": "YYYY-MM-DD or null (null = unknown, see to_verify)",
    "obligation": "string, the rule in ONE sentence",
    "caveat": "string, fine print we must not lose (may be empty)",
    "test_type": TEST_TYPES,
    "threshold": "number or null",
    "to_verify": "list of strings, things not double-checked (may be empty)",
    "applicability_uncertain": "bool; true forces the label 'Needs human review'",
    "retrieved_on": "YYYY-MM-DD, when the source file was saved",
}


def _is_date(value):
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def resolve_source(patterns, sources_dir="sources"):
    """Try each glob pattern in order; return (pattern_that_worked, [files]).

    Patterns are fallbacks because saved filenames vary. On Windows the match
    is case-insensitive.
    """
    for pattern in patterns:
        hits = sorted({p for p in Path(sources_dir).rglob(pattern) if p.is_file()})
        if hits:
            return pattern, hits
    return None, []


def validate(card, sources_dir="sources"):
    """Return (errors, matched_files). Errors are plain-English strings."""
    errors = []
    for field in REQUIRED:
        if field not in card:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors, []

    if not ID_RE.match(str(card["id"])):
        errors.append("id must look like US-ECOA-PROHIBITED")
    if card["jurisdiction"] not in JURISDICTIONS:
        errors.append(f"jurisdiction must be one of {JURISDICTIONS}")
    if card["status"] not in STATUSES:
        errors.append(f"status must be one of {STATUSES}")

    for field, allowed in (("applies_to", TOPICS), ("test_type", TEST_TYPES)):
        value = card[field]
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
        elif any(v not in allowed for v in value):
            errors.append(f"{field} has an unknown value; allowed: {allowed}")

    if not str(card["source_url"]).startswith("http"):
        errors.append("source_url must be a link starting with http")
    if not isinstance(card["source_file"], list) or not card["source_file"]:
        errors.append("source_file must be a non-empty list of glob patterns")

    if card["effective_from"] is not None and not _is_date(card["effective_from"]):
        errors.append("effective_from must be YYYY-MM-DD or null")
    if not _is_date(card["retrieved_on"]):
        errors.append("retrieved_on must be YYYY-MM-DD")
    if card["threshold"] is not None and not isinstance(card["threshold"], (int, float)):
        errors.append("threshold must be a number or null")
    if not isinstance(card["to_verify"], list):
        errors.append("to_verify must be a list (empty is fine)")
    if not isinstance(card["applicability_uncertain"], bool):
        errors.append("applicability_uncertain must be true or false")
    if not str(card["obligation"]).strip():
        errors.append("obligation is empty")
    if len(str(card["obligation"])) > 400:
        errors.append("obligation is too long; keep it to one sentence")

    # An unknown start date is only acceptable if we flagged it somewhere.
    if card["effective_from"] is None and not (card["to_verify"] or
                                               card["applicability_uncertain"]):
        errors.append("effective_from is null but nothing is marked to_verify")

    matched = []
    if isinstance(card["source_file"], list) and card["source_file"]:
        _, matched = resolve_source(card["source_file"], sources_dir)
        if not matched:
            errors.append(f"SOURCE MISSING: no file in {sources_dir}/ matches "
                          f"{card['source_file']}")
    return errors, matched


def load_cards(cards_dir="cards"):
    """Read every card JSON, sorted by id."""
    cards = []
    for path in sorted(Path(cards_dir).glob("*.json")):
        if path.name == "schema.json":
            continue
        cards.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(cards, key=lambda c: c["id"])