"""
Step 7d: Verifier - checks if the LLM followed the rules.
Plain words: Reads the generated report and checks:
1. Does every cited card ID exist in cards/?
2. Does the report respect the label (e.g., no "violate" near a "Not yet in force" card)?
3. Does the card's obligation sentence actually appear in the source file?
Returns pass/fail for each check.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from card_schema import load_cards

# --- evidence-check settings ---------------------------------------------
MATCH_THRESHOLD = 0.60   # fraction of obligation keywords that must be locatable


@lru_cache(maxsize=64)
def _source_text(path_str):
    """Plain text of one source file.

    PDFs are detected by their first bytes (%PDF), NOT by the extension, because
    one saved file has a hash name with no extension. Everything else is read as
    text. Errors come back as a marker string instead of crashing the check.
    """
    path = Path(path_str)
    try:
        with path.open("rb") as fh:
            head = fh.read(5)
    except Exception as exc:
        return f"[READ ERROR: {exc}]"

    if head.startswith(b"%PDF"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[READ ERROR: pypdf not installed. Run: pip install pypdf]"
        try:
            reader = PdfReader(path_str)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return "[READ ERROR: encrypted PDF]"
            chunks = []
            for page in reader.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    chunks.append("")
            return "\n".join(chunks)
        except Exception as exc:
            return f"[READ ERROR: {exc}]"

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"[READ ERROR: {exc}]"


def _normalize(text):
    """Lowercase, repair words split across lines, collapse whitespace.

    PDF extractors break words like 'creditwor-\nthiness'. Joining them first
    stops false 'missing keyword' results.
    """
    t = text.lower()
    for bad, good in (("\u2019", "'"), ("\u2018", "'"), ("\u2013", "-"),
                      ("\u2014", "-"), ("\u00a0", " ")):
        t = t.replace(bad, good)
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _keywords(obligation, limit=12):
    """Distinctive words (length >= 4) from a card's obligation, in order."""
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9][a-z0-9'\-/]*", obligation.lower()):
        if len(w) < 4 or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out

def verify_report(report_text, checked_cards):
    """
    report_text: the LLM-generated paragraph
    checked_cards: list of {"card": {...}, "label": "...", "reason": "..."}
    Returns a dict with pass/fail for each check.
    """
    results = {
        "citation_check": [],
        "label_respect_check": [],
        "evidence_check": [],
        "overall_pass": True
    }

    # 1. Citation check: every [CARD-ID] in the report must exist
    cited_ids = re.findall(r'\[([A-Z]{2}-[A-Z0-9-]+)\]', report_text)
    all_card_ids = {c["id"] for c in load_cards()}

    for card_id in cited_ids:
        exists = card_id in all_card_ids
        results["citation_check"].append({
            "card_id": card_id,
            "exists": exists,
            "pass": exists
        })
        if not exists:
            results["overall_pass"] = False

    # 2. Label respect check: no "violate" near a "Not yet in force" or "Heuristic only" card
    for item in checked_cards:
        card_id = item["card"]["id"]
        label = item["label"]

        pattern = rf'\[{re.escape(card_id)}\][^.]*\.'
        matches = re.findall(pattern, report_text, re.IGNORECASE)

        for sentence in matches:
            has_violate = bool(re.search(r'\b(violat|breach|infring|illegal)\b',
                                         sentence, re.IGNORECASE))
            if label in ["Not yet in force", "Heuristic only", "Out of scope"] and has_violate:
                results["label_respect_check"].append({
                    "card_id": card_id, "label": label,
                    "sentence": sentence.strip(), "pass": False
                })
                results["overall_pass"] = False
            else:
                results["label_respect_check"].append({
                    "card_id": card_id, "label": label, "pass": True
                })

    # 3. Evidence check: are the card's obligation keywords in the source file?
    for item in checked_cards:
        card = item["card"]
        card_id = card["id"]
        kws = _keywords(card["obligation"])

        best = {"ratio": 0.0, "file": None, "matched": [], "missing": list(kws)}
        read_note = ""

        for pattern in card.get("source_file", []):
            for hit in sorted(Path("sources").rglob(pattern)):
                if not hit.is_file():
                    continue
                raw = _source_text(str(hit))
                if raw.startswith("[READ ERROR"):
                    read_note = raw.strip("[]")
                    continue
                text = _normalize(raw)
                matched = [k for k in kws if k in text]
                ratio = len(matched) / len(kws) if kws else 0.0
                if ratio > best["ratio"]:
                    best = {"ratio": ratio, "file": hit.name, "matched": matched,
                            "missing": [k for k in kws if k not in matched]}
                if best["ratio"] >= 1.0:
                    break
            if best["ratio"] >= 1.0:
                break

        passed = bool(kws) and best["ratio"] >= MATCH_THRESHOLD
        entry = {
            "card_id": card_id,
            "found_in_source": passed,
            "match_ratio": round(best["ratio"], 3),
            "matched_keywords": best["matched"],
            "missing_keywords": best["missing"],
            "source_file": best["file"],
            "pass": passed,
        }
        if read_note and not passed:
            entry["note"] = read_note
        if not kws:
            entry["note"] = "obligation has no words of 4+ characters"

        results["evidence_check"].append(entry)
        if not passed:
            results["overall_pass"] = False

    return results


def main():
    print("Testing Verifier...\n")
    
    # Mock a report and checked cards
    report = """The under‑25 cohort shows a selection‑rate ratio of 0.77, which is statistically significant (p = 0.0004). This rule will apply from 2 December 2027, so it is not a violation today [EU-AIA-ART10]. Note: This audit is based on synthetic or proxy data where applicable, and selection-rate gaps may reflect legitimate risk differences."""
    
    checked_cards = [
        {
            "card": {
                "id": "EU-AIA-ART10",
                "obligation": "Providers of high-risk AI systems must examine training, validation and testing data for possible biases and address data gaps or shortcomings.",
                "source_file": ["*2024_1689*", "*2024*1689*"]
            },
            "label": "Not yet in force",
            "reason": "Rule starts 2027-12-02, audit date is 2026-11-24"
        }
    ]

    checked_cards.append({
        "card": {
            "id": "IN-FREEAI-SUTRA4",
            "obligation": ("Regulated entities should ensure their AI systems "
                           "treat customers fairly and equitably."),
            "source_file": ["*FREE*AI*", "*FREEAI*"],
        },
        "label": "Applies",
        "reason": "guidance, no date gate",
    })
    
    results = verify_report(report, checked_cards)
    
    print("CITATION CHECK:")
    for item in results["citation_check"]:
        status = "PASS" if item["pass"] else "FAIL"
        print(f"  [{item['card_id']}] exists: {status}")
    
    print("\nLABEL RESPECT CHECK:")
    for item in results["label_respect_check"]:
        status = "PASS" if item["pass"] else "FAIL"
        print(f"  [{item['card_id']}] label={item['label']}: {status}")
    
        print("\nEVIDENCE CHECK:")
    for item in results["evidence_check"]:
        status = "PASS" if item["pass"] else "FAIL"
        src = item["source_file"] or "no file matched"
        print(f"  [{item['card_id']}] match {item['match_ratio']:.2f} "
              f"in {src}: {status}")
        if item["missing_keywords"]:
            print(f"        missing: {', '.join(item['missing_keywords'][:6])}")
        if item.get("note"):
            print(f"        note: {item['note']}")
    
    # Save results
    Path("reports").mkdir(exist_ok=True)
    with open("reports/verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to reports/verification_results.json")


if __name__ == "__main__":
    main()