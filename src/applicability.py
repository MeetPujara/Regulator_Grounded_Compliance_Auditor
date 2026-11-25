"""
Step 7b: Applicability Checker.
Plain words: Pure code that asks 3 questions about a retrieved card:
1. Right country? 2. Right topic? 3. Active on the audit date?
Returns one of 6 labels. NO LLM is used here.
"""

def check_applicability(card, finding_jurisdiction, finding_topic, audit_date_str):
    # 1. Jurisdiction check
    if card["jurisdiction"] != finding_jurisdiction:
        return "Wrong jurisdiction", f"Card is {card['jurisdiction']}, finding is {finding_jurisdiction}"
    
    # 2. Topic check
    if finding_topic not in card["applies_to"]:
        return "Out of scope", f"Card applies to {card['applies_to']}, finding is {finding_topic}"
        
    # 3. Uncertainty check (forces human review)
    if card.get("applicability_uncertain", False):
        return "Needs human review", "Card has applicability_uncertain=True"
        
    # 4. Date check
    eff_date = card.get("effective_from")
    if eff_date is None:
        return "Needs human review", "effective_from is null"
        
    if eff_date > audit_date_str:
        return "Not yet in force", f"Rule starts {eff_date}, audit date is {audit_date_str}"
        
    # 5. Status check
    if card["status"] == "heuristic":
        return "Heuristic only", "Rule is a heuristic, not binding law"
        
    # If all checks pass
    return "Applies", "All checks passed"