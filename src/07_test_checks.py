"""
Step 7: Test Retrieval and Applicability Checker.
Run from project root: python src\07_test_checks.py
"""
from retrieval import retrieve_cards
from applicability import check_applicability

AUDIT_DATE = "2026-11-24"

def main():
    print("Testing Retrieval & Applicability Checker...\n")
    
    # Example 1: EU age finding (German dataset maps to EU jurisdiction)
    finding_jurisdiction = "EU"
    finding_topic = "lending"
    finding_text = "under 25 age group has selection rate ratio 0.77, significant gap"
    finding_test_type = "ratio_test"
    
    print(f"Finding: {finding_text}")
    print(f"Jurisdiction: {finding_jurisdiction}, Topic: {finding_topic}, Test: {finding_test_type}\n")
    
    top_cards = retrieve_cards(finding_jurisdiction, finding_test_type, finding_text, top_k=3)
    
    print(f"Retrieved {len(top_cards)} cards:")
    for i, card in enumerate(top_cards, 1):
        label, reason = check_applicability(card, finding_jurisdiction, finding_topic, AUDIT_DATE)
        print(f"{i}. [{card['id']}] {card['clause']}")
        print(f"   Status: {card['status']} | Effective: {card['effective_from']}")
        print(f"   -> LABEL: {label.upper()}")
        print(f"   Reason: {reason}\n")

    # Example 2: US sex finding
    print("-" * 60)
    finding_jurisdiction = "US"
    finding_text = "male applicants approved at 0.90 rate vs female 1.0, significant"
    finding_test_type = "ratio_test"
    
    print(f"Finding: {finding_text}")
    print(f"Jurisdiction: {finding_jurisdiction}, Topic: {finding_topic}, Test: {finding_test_type}\n")
    
    top_cards = retrieve_cards(finding_jurisdiction, finding_test_type, finding_text, top_k=3)
    
    print(f"Retrieved {len(top_cards)} cards:")
    for i, card in enumerate(top_cards, 1):
        label, reason = check_applicability(card, finding_jurisdiction, finding_topic, AUDIT_DATE)
        print(f"{i}. [{card['id']}] {card['clause']}")
        print(f"   Status: {card['status']} | Effective: {card['effective_from']}")
        print(f"   -> LABEL: {label.upper()}")
        print(f"   Reason: {reason}\n")

if __name__ == "__main__":
    main()