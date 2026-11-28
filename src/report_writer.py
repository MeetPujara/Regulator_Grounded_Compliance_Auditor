"""
Step 7c: Report Writer using Groq LLM.
Plain words: The LLM writes the plain-English audit report. It is given the 
finding numbers and the CHECKED cards with their labels. It MUST cite card IDs, 
obey the labels (e.g., never say "violation" for a heuristic), and never invent 
facts. Temperature is 0 to minimize creativity.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

SYSTEM_PROMPT = """You are a regulatory compliance auditor. Your job is to write a 
short, plain-English explanation of a fairness finding for a non-technical bank manager.

RULES:
1. Use ONLY the provided cards. Do not invent laws, dates, or penalties.
2. Cite card IDs inline like this: [US-REGB-DISPARATE-IMPACT].
3. OBEY the label exactly:
   - If "Not yet in force": Say "This rule will apply from [date], so it is not a violation today."
   - If "Heuristic only": Say "This is a screening indicator borrowed from employment law, not a legal violation."
   - If "Out of scope": Say "This rule does not apply to lending."
   - If "Needs human review": Say "A human lawyer must verify this, as [reason]."
   - If "Applies": State the obligation clearly.
4. Keep it to 1 short paragraph (3-4 sentences max).
5. Always end with this exact limitations sentence: "Note: This audit is based on synthetic or proxy data where applicable, and selection-rate gaps may reflect legitimate risk differences."
"""

def write_report(finding_text, checked_cards):
    """
    finding_text: e.g., "Under-25s approved at 0.77x the rate of older applicants (p=0.0004)."
    checked_cards: list of dicts like {"card": {...}, "label": "Not yet in force", "reason": "..."}
    """
    # Format the cards for the LLM
    cards_text = ""
    for i, item in enumerate(checked_cards, 1):
        c = item["card"]
        cards_text += f"Card {i} ID: {c['id']}\n"
        cards_text += f"Label: {item['label'].upper()}\n"
        cards_text += f"Reason: {item['reason']}\n"
        cards_text += f"Obligation: {c['obligation']}\n"
        cards_text += f"Caveat: {c['caveat']}\n\n"

    user_prompt = f"""
    FINDING: {finding_text}
    
    APPLICABLE RULE CARDS:
    {cards_text}
    
    Write the audit report paragraph following all rules.
    """

    client = _get_client()
    if client is None:
        return "[ERROR: GROQ_API_KEY is not set in .env.]"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # Fast, free-tier Groq model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: Groq API failed. Check your GROQ_API_KEY in .env. Details: {e}]"


def main():
    print("Testing Report Writer...\n")
    
    # Mock a finding and the checked cards from Step 7b
    finding = "Under-25 age group has a selection rate ratio of 0.77 compared to the best group, with a significant p-value of 0.0004."
    
    checked_cards = [
        {
            "card": {
                "id": "EU-AIA-ART10",
                "obligation": "Providers of high-risk AI systems must examine training, validation and testing data for possible biases and address data gaps or shortcomings.",
                "caveat": "Duty on providers of high-risk systems. For Annex III credit-scoring systems the obligations apply from 2 December 2027."
            },
            "label": "Not yet in force",
            "reason": "Rule starts 2027-12-02, audit date is 2026-11-24"
        }
    ]
    
    print("FINDING:", finding)
    print("\nGENERATED REPORT:")
    print("-" * 60)
    report = write_report(finding, checked_cards)
    print(report)
    print("-" * 60)

if __name__ == "__main__":
    main()