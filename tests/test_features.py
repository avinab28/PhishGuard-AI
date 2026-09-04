import pytest
from backend.services.feature_extraction import (
    URL_FEATURE_NAMES,
    extract_url_features,
    url_features_to_vector,
    clean_message_text,
    calculate_entropy
)

def test_url_features_keys():
    url = "https://www.google.com/search?q=cybersecurity"
    feats = extract_url_features(url)
    assert isinstance(feats, dict)
    for name in URL_FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"
    assert feats["is_https"] == 1.0
    assert feats["has_ip"] == 0.0

def test_url_features_ip_detection():
    url = "http://192.168.1.100:8080/login.php"
    feats = extract_url_features(url)
    assert feats["has_ip"] == 1.0
    assert feats["is_https"] == 0.0
    assert feats["has_port"] == 1.0
    assert feats["suspicious_keywords_count"] >= 1.0

def test_url_features_vector_conversion():
    url = "http://phishing-update-paypal.xyz/account/login"
    feats = extract_url_features(url)
    vec = url_features_to_vector(feats)
    assert len(vec) == len(URL_FEATURE_NAMES)
    assert isinstance(vec[0], float)
    assert feats["suspicious_tld"] == 1.0

def test_calculate_entropy():
    # Constant string has 0 entropy
    assert calculate_entropy("aaaaaaa") == 0.0
    # Random string has high entropy
    e_high = calculate_entropy("x9k2m8q4z1p0")
    assert e_high > 3.0

def test_clean_message_text():
    raw = "URGENT! Call 55512345 or visit https://scam.xyz/claim now to get $500 cash!"
    cleaned = clean_message_text(raw)
    assert "<url>" in cleaned or "<URL>" in cleaned
    assert "<money>" in cleaned or "<MONEY>" in cleaned
    assert "urgent" in cleaned
    assert "!" not in cleaned
