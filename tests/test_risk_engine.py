import pytest
from backend.services.risk_engine import evaluate_risk
from backend.services.explanation_engine import (
    analyze_url_heuristics,
    analyze_message_heuristics
)
from backend.services.feature_extraction import extract_url_features

def test_risk_evaluation_tiers():
    # Low: < 0.40
    level, verdict, conf, action = evaluate_risk(0.15, "url")
    assert level == "LOW"
    assert conf == 85.0
    assert "Low Risk" in verdict

    # Boundary Low: 0.39
    level, _, _, _ = evaluate_risk(0.39, "url")
    assert level == "LOW"

    # Medium: 0.40 - 0.69
    level, verdict, conf, _ = evaluate_risk(0.45, "url")
    assert level == "MEDIUM"
    assert conf == 55.0

    level, _, _, _ = evaluate_risk(0.69, "message")
    assert level == "MEDIUM"

    # High: >= 0.70
    level, verdict, conf, _ = evaluate_risk(0.70, "url")
    assert level == "HIGH"
    assert conf == 70.0

    level, _, conf, _ = evaluate_risk(0.98, "message")
    assert level == "HIGH"
    assert conf == 98.0

def test_risk_evaluation_clamping():
    level_neg, _, conf_neg, _ = evaluate_risk(-0.5, "url")
    assert level_neg == "LOW"
    assert conf_neg == 100.0

    level_over, _, conf_over, _ = evaluate_risk(1.5, "url")
    assert level_over == "HIGH"
    assert conf_over == 100.0

def test_url_heuristic_indicators():
    url = "http://192.168.1.1/admin/login?user=test@victim"
    feats = extract_url_features(url)
    indicators = analyze_url_heuristics(url, feats)
    names = [i.name for i in indicators]
    assert "IP Address Used as Hostname" in names
    assert "Insecure HTTP Protocol" in names
    assert "Embedded Userinfo '@' Symbol" in names

def test_message_heuristic_indicators():
    msg = "URGENT: Your bank account is suspended! Confirm your password and PIN immediately."
    indicators = analyze_message_heuristics(msg)
    names = [i.name for i in indicators]
    assert any("Urgency" in n for n in names)
    assert any("Credential" in n for n in names)
