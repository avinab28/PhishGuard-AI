from typing import Tuple, Literal
from backend.config import settings

RiskTier = Literal["LOW", "MEDIUM", "HIGH"]

def evaluate_risk(probability: float, scan_type: str = "url") -> Tuple[RiskTier, str, float, str]:
    """
    Evaluates probability according to PhishGuard-AI guidelines:
    - LOW: probability < 0.40
    - MEDIUM: 0.40 <= probability < 0.70
    - HIGH: probability >= 0.70

    Returns:
    - risk_level (LOW, MEDIUM, HIGH)
    - verdict (descriptive summary)
    - confidence_score (0.0 to 100.0)
    - recommended_action (actionable defense guidance)
    """
    # Clamp probability to valid range [0.0, 1.0]
    prob = max(0.0, min(1.0, float(probability)))
    
    # Calculate confidence score: distance from uncertainty (0.5) scaled to 0-100%
    # or certainty in the assigned classification
    if prob >= 0.5:
        confidence = round(prob * 100.0, 2)
    else:
        confidence = round((1.0 - prob) * 100.0, 2)

    if prob < settings.RISK_THRESHOLD_LOW:
        risk_level: RiskTier = "LOW"
        if scan_type == "url":
            verdict = "Low Risk - Structural patterns align with legitimate websites."
            recommended_action = "Standard caution advised. Always verify the domain in the browser address bar before entering credentials."
        else:
            verdict = "Low Risk - No typical scam or phishing patterns detected."
            recommended_action = "Message appears benign. Maintain general vigilance regarding unsolicited communications."

    elif prob < settings.RISK_THRESHOLD_HIGH:
        risk_level = "MEDIUM"
        if scan_type == "url":
            verdict = "Moderate Risk - Suspicious structural traits or unusual parameters observed."
            recommended_action = "Exercise caution. Do NOT enter passwords, financial info, or download attachments from this URL."
        else:
            verdict = "Moderate Risk - Potential social engineering or persuasion tactics detected."
            recommended_action = "Do not click links or reply with sensitive information. Verify the sender through a secondary channel."

    else:
        risk_level = "HIGH"
        if scan_type == "url":
            verdict = "High Risk - Strong indicators of a phishing or deceptive domain."
            recommended_action = "CRITICAL: Do NOT visit this link or interact with the page. Block and report this URL immediately."
        else:
            verdict = "High Risk - Heavy urgency, credential harvesting, or fraudulent patterns identified."
            recommended_action = "CRITICAL: Phishing/Smishing threat. Delete immediately. Never share OTPs, passwords, or personal data."

    return risk_level, verdict, confidence, recommended_action
