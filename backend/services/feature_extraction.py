import math
import re
from urllib.parse import urlparse
from typing import Dict, Any, List

# Standardized ordered list of feature keys for ANN training & inference
URL_FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_underscores",
    "count_slashes",
    "count_at",
    "count_question",
    "count_percent",
    "count_equal",
    "count_digits",
    "digits_ratio",
    "is_https",
    "has_ip",
    "subdomain_count",
    "suspicious_keywords_count",
    "suspicious_tld",
    "domain_entropy",
    "has_punycode",
    "has_port",
    "double_slash_in_path"
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "verification", "secure", "security", "update",
    "account", "banking", "bank", "signin", "sign-in", "confirm",
    "password", "credential", "wallet", "support", "billing", "authenticate",
    "service", "recover", "claim", "free", "gift", "bonus", "prize",
    "urgent", "alert", "validate", "suspended", "limited", "unlock"
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "fit", "work",
    "icu", "cyou", "vip", "monster", "rest", "bar", "live", "link"
}

IP_REGEX = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)

def calculate_entropy(text: str) -> float:
    """Calculates the Shannon entropy of a string."""
    if not text:
        return 0.0
    length = len(text)
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    for count in freq.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    return round(entropy, 4)

def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extracts purely static, local structural features from a URL.
    STRICT SECURITY RULE: Zero external network/DNS calls.
    """
    url_str = url.strip()
    
    # Normalize scheme for parsing if absent
    raw_parse = urlparse(url_str if "://" in url_str else f"http://{url_str}")
    hostname = (raw_parse.hostname or "").lower()
    path = raw_parse.path or ""
    scheme = (raw_parse.scheme or "").lower()
    port = raw_parse.port

    # Length features
    url_len = len(url_str)
    host_len = len(hostname)
    path_len = len(path)

    # Character counts in the full URL
    c_dots = url_str.count(".")
    c_hyphens = url_str.count("-")
    c_underscores = url_str.count("_")
    c_slashes = url_str.count("/")
    c_at = url_str.count("@")
    c_question = url_str.count("?")
    c_percent = url_str.count("%")
    c_equal = url_str.count("=")
    c_digits = sum(c.isdigit() for c in url_str)
    digits_ratio = round(c_digits / max(url_len, 1), 4)

    # Protocol & host checks
    is_https = 1.0 if scheme == "https" else 0.0
    has_ip = 1.0 if IP_REGEX.match(hostname) else 0.0
    
    # Subdomain calculation
    parts = hostname.split(".")
    # If standard domain like example.com -> 2 parts -> 0 subdomains
    # If sub.example.com -> 3 parts -> 1 subdomain
    subdomain_count = float(max(0, len(parts) - 2)) if not has_ip else 0.0

    # Suspicious keywords presence
    url_lower = url_str.lower()
    kw_count = float(sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower))

    # TLD analysis
    tld = parts[-1] if len(parts) > 1 and not has_ip else ""
    suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0

    # Domain entropy
    domain_entropy = calculate_entropy(hostname)

    # Specific phishing evasion indicators
    has_punycode = 1.0 if "xn--" in hostname else 0.0
    has_port = 1.0 if port is not None and port not in (80, 443) else 0.0
    double_slash_path = 1.0 if "//" in path else 0.0

    features = {
        "url_length": float(url_len),
        "hostname_length": float(host_len),
        "path_length": float(path_len),
        "count_dots": float(c_dots),
        "count_hyphens": float(c_hyphens),
        "count_underscores": float(c_underscores),
        "count_slashes": float(c_slashes),
        "count_at": float(c_at),
        "count_question": float(c_question),
        "count_percent": float(c_percent),
        "count_equal": float(c_equal),
        "count_digits": float(c_digits),
        "digits_ratio": float(digits_ratio),
        "is_https": float(is_https),
        "has_ip": float(has_ip),
        "subdomain_count": float(subdomain_count),
        "suspicious_keywords_count": float(kw_count),
        "suspicious_tld": float(suspicious_tld),
        "domain_entropy": float(domain_entropy),
        "has_punycode": float(has_punycode),
        "has_port": float(has_port),
        "double_slash_in_path": float(double_slash_path),
    }
    return features

def url_features_to_vector(features: Dict[str, Any]) -> List[float]:
    """Converts feature dictionary into ordered vector matching URL_FEATURE_NAMES."""
    return [features[name] for name in URL_FEATURE_NAMES]

def clean_message_text(text: str) -> str:
    """
    Cleans and standardizes message text for RNN tokenization.
    - Lowercases text
    - Replaces URLs with placeholder
    - Removes punctuation while preserving whitespace
    - Normalizes multiple spaces
    """
    if not text:
        return ""
    
    # Lowercase
    cleaned = text.lower()
    
    # Replace URLs with special token
    cleaned = re.sub(r"https?://\S+|www\.\S+", " <URL> ", cleaned)
    
    # Replace email addresses with special token
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " <EMAIL> ", cleaned)
    
    # Replace phone numbers / long digit sequences with special token
    cleaned = re.sub(r"\b\d{5,}\b", " <NUMBER> ", cleaned)
    
    # Replace dollar/currency amounts
    cleaned = re.sub(r"[$€£]\d+(?:\.\d+)?", " <MONEY> ", cleaned)
    
    # Remove punctuation except placeholder brackets
    cleaned = re.sub(r"[^\w\s<>]", " ", cleaned)
    
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned
