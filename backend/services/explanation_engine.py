import re
from typing import List, Dict, Any
from urllib.parse import urlparse
from backend.schemas import HeuristicIndicator
from backend.services.feature_extraction import (
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_TLDS,
    IP_REGEX,
    calculate_entropy
)

URGENCY_PATTERNS = [
    r"\b(urgent|immediately|action required|suspended|suspension|locked|warning|terminated|within 24 hours|deadline|expires)\b",
    r"\b(unauthorized|compromised|breach|security alert|fraud detected)\b"
]

CREDENTIAL_PATTERNS = [
    r"\b(password|passcode|pin|otp|login credentials|verify your account|confirm your identity|update billing)\b",
    r"\b(click here to verify|validate your details|security question|ssn|social security)\b"
]

FINANCIAL_LURE_PATTERNS = [
    r"\b(winner|won|congratulations|claim your prize|lottery|cash reward|free gift|refund waiting|crypto bonus)\b",
    r"\b(\$\d+|\£\d+|\€\d+|100% free|guaranteed return)\b"
]

IMPERSONATION_PATTERNS = [
    r"\b(bank of america|wells fargo|chase|paypal|apple support|microsoft support|amazon security|irs|fedex delivery|usps tracking)\b"
]

SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "buff.ly", "ow.ly", "goo.gl", "cutt.ly"}

def analyze_url_heuristics(url: str, features: Dict[str, Any]) -> List[HeuristicIndicator]:
    """
    Evaluates rule-based heuristic threat indicators on the URL.
    Outputs clear explanations decoupled from the neural network probability.
    """
    indicators: List[HeuristicIndicator] = []
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # 1. IP Address usage
    if features.get("has_ip", 0.0) == 1.0:
        indicators.append(HeuristicIndicator(
            name="IP Address Used as Hostname",
            description="The URL uses a raw IP address instead of a registered domain name, a classic technique to bypass reputation filters.",
            severity="CRITICAL",
            triggered=True,
            details=f"Host resolved directly to: {hostname}"
        ))

    # 2. Credential obfuscation via '@'
    if features.get("count_at", 0.0) > 0:
        indicators.append(HeuristicIndicator(
            name="Embedded Userinfo '@' Symbol",
            description="URLs containing '@' can trick browsers into ignoring preceding text, obfuscating the actual destination host.",
            severity="CRITICAL",
            triggered=True,
            details=f"Found {int(features.get('count_at', 1))} '@' symbol(s)"
        ))

    # 3. Protocol Security
    if features.get("is_https", 1.0) == 0.0:
        indicators.append(HeuristicIndicator(
            name="Insecure HTTP Protocol",
            description="The URL does not use TLS/HTTPS encryption. Any data transmitted can be intercepted in plain text.",
            severity="WARNING",
            triggered=True,
            details="Scheme: http://"
        ))

    # 4. Abnormal URL Length
    url_len = int(features.get("url_length", 0))
    if url_len > 75:
        indicators.append(HeuristicIndicator(
            name="Excessive URL Length",
            description="Phishing URLs frequently use lengthy strings to conceal targets, pack payloads, or mimic legitimate paths.",
            severity="WARNING",
            triggered=True,
            details=f"Total length: {url_len} characters (typical legitimate URLs are < 50)"
        ))

    # 5. Suspicious Keywords
    kw_count = int(features.get("suspicious_keywords_count", 0))
    if kw_count > 0:
        found_kw = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()]
        indicators.append(HeuristicIndicator(
            name="Authentication & Security Keywords",
            description="Target contains words associated with high-value security, billing, or login workflows.",
            severity="WARNING" if kw_count < 2 else "CRITICAL",
            triggered=True,
            details=f"Detected keywords: {', '.join(found_kw[:5])}"
        ))

    # 6. High Subdomain Count
    sub_count = int(features.get("subdomain_count", 0))
    if sub_count >= 2:
        indicators.append(HeuristicIndicator(
            name="Multiple Subdomain Levels",
            description="Excessive subdomains are frequently engineered to mimic trusted organizations (e.g., 'paypal.com.attacker.com').",
            severity="WARNING",
            triggered=True,
            details=f"{sub_count} subdomain tiers found"
        ))

    # 7. Suspicious TLD
    if features.get("suspicious_tld", 0.0) == 1.0:
        tld = hostname.split(".")[-1] if "." in hostname else ""
        indicators.append(HeuristicIndicator(
            name="High-Risk Top-Level Domain (TLD)",
            description=f"The domain uses '{tld}', a TLD with statistically elevated rates of abuse, spam, and disposable hosting.",
            severity="WARNING",
            triggered=True,
            details=f"TLD: .{tld}"
        ))

    # 8. High Domain Entropy
    entropy = features.get("domain_entropy", 0.0)
    if entropy > 3.6:
        indicators.append(HeuristicIndicator(
            name="High Domain Randomness / Entropy",
            description="Domain name exhibits high algorithmic character entropy, characteristic of Domain Generation Algorithms (DGA) or disposable throwaway domains.",
            severity="WARNING",
            triggered=True,
            details=f"Calculated Shannon entropy: {entropy:.2f} bits/char"
        ))

    # 9. Punycode / Homograph Attack
    if features.get("has_punycode", 0.0) == 1.0:
        indicators.append(HeuristicIndicator(
            name="Punycode Homograph Candidate",
            description="Domain contains 'xn--', which can be used to render Cyrillic or Greek lookalike characters that impersonate real brands.",
            severity="CRITICAL",
            triggered=True,
            details="Punycode prefix detected"
        ))

    # 10. Redirect slash markers
    if features.get("double_slash_in_path", 0.0) == 1.0:
        indicators.append(HeuristicIndicator(
            name="Path Redirection Marker '//'",
            description="Double slashes inside the path can be used for open redirects or confusing web server path parsers.",
            severity="WARNING",
            triggered=True,
            details="Path contains '//'"
        ))

    # If no negative indicators triggered, note clean static profile
    if not indicators:
        indicators.append(HeuristicIndicator(
            name="Clean Static Profile",
            description="Static heuristic inspection found no abnormal keywords, protocol irregularities, or obfuscation symbols.",
            severity="INFO",
            triggered=True,
            details="Standard structural conventions observed"
        ))

    return indicators

def analyze_message_heuristics(message: str) -> List[HeuristicIndicator]:
    """
    Evaluates rule-based heuristic threat indicators on message texts (SMS/Email/Chat).
    """
    indicators: List[HeuristicIndicator] = []
    text_lower = message.lower()

    # 1. Urgency & Coercion
    urgency_matches = []
    for pat in URGENCY_PATTERNS:
        urgency_matches.extend(re.findall(pat, text_lower))
    if urgency_matches:
        indicators.append(HeuristicIndicator(
            name="Psychological Urgency & Threat Triggers",
            description="Message employs high-pressure tactics or threat of account suspension to bypass critical thinking.",
            severity="CRITICAL",
            triggered=True,
            details=f"Detected trigger terms: {', '.join(set(urgency_matches[:4]))}"
        ))

    # 2. Credential Solicitation
    cred_matches = []
    for pat in CREDENTIAL_PATTERNS:
        cred_matches.extend(re.findall(pat, text_lower))
    if cred_matches:
        indicators.append(HeuristicIndicator(
            name="Credential / Sensitive Data Harvesting",
            description="Message solicits confidential data (passwords, OTPs, PINs, or personal identity verification).",
            severity="CRITICAL",
            triggered=True,
            details=f"Sensitive requests: {', '.join(set(cred_matches[:4]))}"
        ))

    # 3. Financial Lure
    fin_matches = []
    for pat in FINANCIAL_LURE_PATTERNS:
        fin_matches.extend(re.findall(pat, text_lower))
    if fin_matches:
        indicators.append(HeuristicIndicator(
            name="Financial Lure / Prize Baits",
            description="Message offers unsolicited monetary rewards, lottery wins, or unverified refunds.",
            severity="WARNING",
            triggered=True,
            details=f"Lure indicators: {', '.join(set(fin_matches[:4]))}"
        ))

    # 4. Brand Impersonation
    imp_matches = []
    for pat in IMPERSONATION_PATTERNS:
        imp_matches.extend(re.findall(pat, text_lower))
    if imp_matches:
        indicators.append(HeuristicIndicator(
            name="Brand Impersonation Suspect",
            description="Message claims association with a major financial institution, courier, or tech provider.",
            severity="WARNING",
            triggered=True,
            details=f"Referenced entities: {', '.join(set(imp_matches[:3]))}"
        ))

    # 5. Embedded Links / Shorteners
    urls_found = re.findall(r"(https?://\S+|www\.\S+)", message)
    if urls_found:
        has_shortener = any(any(s in u.lower() for s in SHORTENER_DOMAINS) for u in urls_found)
        indicators.append(HeuristicIndicator(
            name="Embedded Link in Unsolicited Message",
            description="Message contains hyperlinked web addresses. Phishing scams rely on links to drive victims to malicious portals.",
            severity="CRITICAL" if has_shortener else "WARNING",
            triggered=True,
            details="Includes URL shortener" if has_shortener else f"{len(urls_found)} link(s) detected"
        ))

    # 6. Excessive Capitalization (Panic Inducer)
    words = message.split()
    caps_words = [w for w in words if len(w) > 3 and w.isupper() and w.isalpha()]
    if len(caps_words) >= 3:
        indicators.append(HeuristicIndicator(
            name="Excessive Capitalization",
            description="Multiple uppercase words designed to simulate alarm or grab urgent attention.",
            severity="INFO",
            triggered=True,
            details=f"Capitalized terms: {', '.join(caps_words[:4])}"
        ))

    if not indicators:
        indicators.append(HeuristicIndicator(
            name="Clean Text Profile",
            description="No overt urgency language, financial lures, or credential requests detected in the text.",
            severity="INFO",
            triggered=True,
            details="Conversational baseline text"
        ))

    return indicators
