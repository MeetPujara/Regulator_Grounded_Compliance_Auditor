"""
RGCA — Regulatory-Grounded Compliance Auditor (AI Loan Auditor)
Canonical Flask application entry point.
Run: python app.py
"""
from web_app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
