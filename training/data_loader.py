import os
import sys
import zipfile
import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Tuple
from sklearn.model_selection import train_test_split
from backend.config import settings

DATA_RAW_DIR = settings.DATA_DIR / "raw"

# Curated synthetic/fallback samples guaranteeing diversity and realistic distributions
SAMPLE_LEGIT_URLS = [
    "https://www.google.com/search?q=cybersecurity+best+practices",
    "https://github.com/torvalds/linux",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://stackoverflow.com/questions/tagged/python",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.microsoft.com/en-us/software-download/windows11",
    "https://aws.amazon.com/ec2/pricing/",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP",
    "https://news.ycombinator.com/item?id=25000000",
    "https://www.nytimes.com/section/technology",
    "https://www.apple.com/iphone-15-pro/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://numpy.org/doc/stable/reference/index.html",
    "https://pandas.pydata.org/docs/getting_started/index.html",
    "https://scikit-learn.org/stable/modules/classes.html",
    "https://www.bbc.com/news/world",
    "https://hub.docker.com/_/python",
    "https://www.reddit.com/r/netsec/",
    "https://www.cloudflare.com/learning/dns/what-is-dns/",
    "https://gitlab.com/explore/projects",
    "https://www.linkedin.com/feed/",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://medium.com/topic/cybersecurity",
    "https://pypi.org/project/tensorflow/",
    "https://www.kaggle.com/datasets",
    "https://www.khanacademy.org/computing",
    "https://archive.org/web/",
    "https://curl.se/docs/manpage.html",
    "https://www.ietf.org/rfc/rfc3986.txt",
    "https://about.gitlab.com/handbook/"
]

SAMPLE_PHISH_URLS = [
    "http://paypal-security-update.account-verification-login.com/auth/login.php",
    "http://192.168.1.105:8080/secure/bankofamerica-login/update.html",
    "http://appleid.apple.com.verify-billing-info.xyz/account/manage",
    "http://chase-bank-alert.service-notification.top/signin?user=client",
    "http://login.microsoftonline.portal-security-auth.tk/common/oauth2",
    "http://wellsfargo.com-security-check.cf/login/authenticate?session=urgent",
    "http://netflix-billing-recovery.card-update.buzz/customer/payment",
    "http://amazon-orders-support.refund-claim.icu/verify/order_id=98712",
    "http://accounts.google.com-recovery-password.rest/ServiceLogin",
    "http://binance.com-login-verify.monster/auth/2fa-reset",
    "http://dhl-package-tracking.delivery-fee-redirection.vip/shipment",
    "http://irs-tax-refund.internal-revenue-claim.work/2024/payment",
    "http://secure-metamask-io.wallet-restore-phrase.xyz/unlock",
    "http://instagram-copyright-infringement.appeal-form.top/verify",
    "http://142.250.190.46/login.php?client=bank&ref=security",
    "http://usps-address-confirm.redelivery-portal.live/tracking/confirm",
    "http://facebook-security-team.checkpoint-case.bar/account/recovery",
    "http://coinbase-auth-ticket.recovery-vault.cyou/signin/challenge",
    "http://citibank-cardholder.card-security-block.fit/online/portal",
    "http://dropbox-shared-document.business-viewer.top/file/view",
    "http://steamcommunity.com-id-trade.trade-offer-steam.xyz/trade/new",
    "http://att-yahoo-mail.account-synchronize.buzz/mail/login",
    "http://ebay-resolution-center.dispute-protection.rest/case",
    "http://gov-stimulus-payment.direct-deposit-portal.icu/form",
    "http://target-giftcard-giveaway.exclusive-reward.top/claim",
    "http://walmart-survey-rewards.cashback-winner.xyz/redeem",
    "http://fidelity-investments-portal.security-verification.work/login",
    "http://adobe-creative-cloud.document-sign-urgent.live/doc/share",
    "http://roblox-free-robux-generator.promo-gift.cf/redeem",
    "http://whatsapp-web-service.qr-authentication.top/session"
]

SAMPLE_HAM_MESSAGES = [
    "Hey, are we still meeting for lunch at the cafeteria at 12:30?",
    "Can you please review the pull request when you get a chance today?",
    "The project report has been uploaded to the shared team drive.",
    "Don't forget Mom's birthday dinner this Friday at 7 PM.",
    "Thanks for sending the lecture notes, they were really helpful!",
    "Could you pick up some milk and bread on your way home tonight?",
    "I'll be running about ten minutes late to the team standup meeting.",
    "Let me know if you want to play tennis this Saturday afternoon.",
    "Where did you leave the office keys yesterday before heading out?",
    "I just finished reading that novel you recommended, it was amazing.",
    "Great work on the presentation yesterday, everyone loved the slides.",
    "Are you free for a quick phone call regarding tomorrow's schedule?",
    "Please find attached the updated invoice for the graphic design work.",
    "We are planning a camping trip next weekend, would you like to join?",
    "Has anyone seen my notebook that was on the conference room table?",
    "The weather forecast says it might rain later, don't forget an umbrella.",
    "Happy anniversary to both of you! Wishing you many more happy years.",
    "I'm at the library studying for finals if you want to join me.",
    "Can you share the contact details of the plumber who fixed your sink?",
    "Dinner was fantastic last night, let's catch up again next week.",
    "Let me know what time suits you best for our interview tomorrow.",
    "I left the car keys on the kitchen counter next to the microwave.",
    "Good morning! Hope you have a productive and relaxing day.",
    "Did you get an email from HR regarding the new holiday schedule?",
    "I'm heading out for a coffee run, does anyone want anything?"
]

SAMPLE_SPAM_MESSAGES = [
    "URGENT: Your Chase Bank account has been suspended! Verify your credentials immediately at http://chase-bank-alert.service-notification.top or your account will be closed.",
    "Congratulations! You've won $5,000 cash in our weekly prize draw! Claim your reward now by texting CLAIM to 88821 or visit http://prize-reward.xyz",
    "SECURITY ALERT: We detected unauthorized login to your Wells Fargo account from an unknown IP. Verify your identity now at http://wellsfargo-security.cf",
    "PayPal Notice: Your payment of $349.99 to Apple Store is pending. If you did not make this purchase, call 1-800-555-0199 or click http://paypal-cancel.top",
    "FINAL WARNING: Your Netflix subscription has expired. Update your billing card within 24 hours to avoid service termination: http://netflix-billing.buzz",
    "IRS Alert: You have an unclaimed tax refund of $1,420.50 waiting. Confirm your SSN and bank details here: http://irs-tax-refund.icu/claim",
    "Your package with tracking #US948194 cannot be delivered due to incomplete address. Update info and pay $1.99 redelivery fee: http://usps-redelivery.live",
    "Amazon Security: Your account is on hold due to suspicious orders. Restore access immediately: http://amazon-account-review.work/login",
    "URGENT: Your Apple ID is locked for security reasons. Enter your passcode to unlock your device: http://appleid-unlock.xyz",
    "You have been pre-approved for a $15,000 personal loan with 0% APR! No credit check required. Apply today: http://quickcash-loans.top",
    "BANK ALERT: Unauthorized debit of $850 detected on your debit card. Text NO to cancel or verify at http://bank-fraud-alert.cf",
    "Hot singles in your area want to chat with you right now! Click here to view profiles: http://chat-singles.monster",
    "ALERT: Your SIM card will be deactivated within 12 hours. Dial *121*88# or verify KYC online: http://telecom-kyc.rest",
    "Crypto Warning: 1.25 BTC withdrawal requested from your Binance account. Not you? Cancel immediately at http://binance-cancel.xyz",
    "DHL Express: Your parcel delivery is pending custom duty payment of $2.50. Pay now: http://dhl-customs-clearance.top",
    "WINNER! Your mobile number won 1st prize of £500,000 in the UK National Lottery. Send name, age, address to claim@lottery-winner.buzz",
    "Security Notification: Google detected unknown device login to your Gmail. Change your password here: http://google-password-reset.top",
    "Dear customer, your credit card loyalty points of 85,000 expire today. Redeem for $500 gift card at http://rewards-redeem.icu",
    "Immediate Action Required: Your vehicle warranty has expired! Renew now to maintain coverage: http://auto-warranty-direct.work",
    "Bank of America: Suspicious activity on card ending in 4102. Confirm your transactions: http://bofa-verify.xyz"
]

def generate_synthetic_url_dataset(n_samples: int = 400) -> pd.DataFrame:
    """
    Generates a balanced, realistic URL dataset for training and verification.
    Combines core high-fidelity samples with varied synthetic permutations.
    """
    records = []
    
    # Base real-world samples
    for u in SAMPLE_LEGIT_URLS:
        records.append({"url": u, "label": 0})
    for u in SAMPLE_PHISH_URLS:
        records.append({"url": u, "label": 1})
        
    # Generate variations
    remaining = max(0, n_samples - len(records))
    legit_domains = [
        "google.com", "github.com", "microsoft.com", "amazon.com", "wikipedia.org",
        "stackoverflow.com", "python.org", "mozilla.org", "nytimes.com", "bbc.com",
        "medium.com", "kaggle.com", "docker.com", "reddit.com", "cloudflare.com"
    ]
    phish_brand_stems = ["paypal", "chase", "appleid", "wellsfargo", "netflix", "binance", "amazon", "coinbase"]
    phish_action_stems = ["login", "verify", "secure", "update", "account-alert", "auth-check", "billing"]
    phish_tlds = ["xyz", "top", "cf", "buzz", "icu", "work", "rest", "monster"]
    
    for i in range(remaining // 2):
        # Synthetic legitimate
        dom = legit_domains[i % len(legit_domains)]
        paths = ["articles", "docs", "help", "community", "products", "tutorials", "explore"]
        p = paths[i % len(paths)]
        sub = f"sub{i}." if i % 4 == 0 else ""
        legit_url = f"https://{sub}{dom}/{p}/{i}?ref=direct&lang=en"
        records.append({"url": legit_url, "label": 0})
        
        # Synthetic phishing
        brand = phish_brand_stems[i % len(phish_brand_stems)]
        act = phish_action_stems[i % len(phish_action_stems)]
        tld = phish_tlds[i % len(phish_tlds)]
        use_ip = (i % 5 == 0)
        if use_ip:
            phish_url = f"http://192.168.{i%255}.{(i*7)%255}:{8000 + i%500}/{brand}-{act}/index.html?token=auth{i}"
        else:
            phish_url = f"http://{brand}-{act}.client-security.{tld}/portal/confirm.php?user_id={i}&session=urgent"
        records.append({"url": phish_url, "label": 1})
        
    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def generate_synthetic_message_dataset(n_samples: int = 400) -> pd.DataFrame:
    """
    Generates a balanced, realistic message dataset for training and verification.
    """
    records = []
    
    for m in SAMPLE_HAM_MESSAGES:
        records.append({"message": m, "label": 0})
    for m in SAMPLE_SPAM_MESSAGES:
        records.append({"message": m, "label": 1})
        
    remaining = max(0, n_samples - len(records))
    
    ham_templates = [
        "Hey {}, are you coming to the {} tonight?",
        "Don't forget to submit the {} before tomorrow morning.",
        "Could you send me the notes from the {} meeting?",
        "Thanks for helping with the {} project yesterday!",
        "I'll be at the {} around {} pm if you want to meet up."
    ]
    ham_fillers = [
        ("Alex", "concert"), ("Sam", "assignment"), ("Chris", "team sync"),
        ("Morgan", "budget"), ("Jordan", "library"), ("Taylor", "dinner")
    ]
    
    spam_templates = [
        "URGENT: Your {} account has been flagged for fraud! Verify your identity at {} or access will be revoked.",
        "Congratulations! You won ${} in our lucky draw! Claim immediately at {}.",
        "FINAL NOTICE: Your {} subscription payment failed. Update your card at {} to avoid interruption.",
        "SECURITY ALERT: Unauthorized password change attempt detected on your {}. Secure your account now at {}.",
        "Package #{}: Redelivery scheduled. Confirm your mailing address and pay ${} fee: {}"
    ]
    spam_entities = [
        ("PayPal", "http://paypal-verification.xyz"),
        ("Bank of America", "http://bofa-security.top"),
        ("Netflix", "http://netflix-billing.buzz"),
        ("Amazon", "http://amazon-account-support.icu"),
        ("Apple ID", "http://apple-login-security.work"),
        ("USPS", "http://usps-parcel-info.live")
    ]
    
    for i in range(remaining // 2):
        # Synthetic ham
        tmpl = ham_templates[i % len(ham_templates)]
        name, item = ham_fillers[i % len(ham_fillers)]
        ham_msg = tmpl.format(name, item)
        records.append({"message": f"{ham_msg} (Ref #{i})", "label": 0})
        
        # Synthetic spam
        s_tmpl = spam_templates[i % len(spam_templates)]
        brand, link = spam_entities[i % len(spam_entities)]
        amt = 1000 + (i * 250) % 9000
        fee = 2.99
        if "draw" in s_tmpl:
            spam_msg = s_tmpl.format(amt, link)
        elif "Package" in s_tmpl:
            spam_msg = s_tmpl.format(f"98{i}42", fee, link)
        else:
            spam_msg = s_tmpl.format(brand, link)
        records.append({"message": f"{spam_msg} [ID: {i}]", "label": 1})
        
    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def fetch_or_generate_url_dataset(use_real_if_available: bool = True) -> pd.DataFrame:
    """
    Attempts to load or fetch real URL datasets.
    Falls back gracefully to the synthetic dataset generator if offline or in quick-build mode.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_RAW_DIR / "urls.csv"
    
    if local_path.exists():
        print(f"Loading cached URL dataset from {local_path}...")
        return pd.read_csv(local_path)
    
    if use_real_if_available:
        try:
            print("Attempting to fetch URL dataset from public curated source...")
            url = "https://raw.githubusercontent.com/datasets/phishing-urls/master/data/urls.csv"
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                if "url" in df.columns and "label" in df.columns:
                    df.to_csv(local_path, index=False)
                    print(f"Downloaded and cached {len(df)} URL samples to {local_path}")
                    return df
        except Exception as e:
            print(f"Real URL dataset download skipped/failed ({e}). Utilizing fallback synthetic generator.")
            
    print("Generating representative synthetic URL dataset (300+ samples)...")
    df = generate_synthetic_url_dataset(n_samples=360)
    df.to_csv(local_path, index=False)
    return df

def fetch_or_generate_message_dataset(use_real_if_available: bool = True) -> pd.DataFrame:
    """
    Attempts to download and parse UCI SMS Spam Collection.
    Falls back gracefully to the synthetic dataset generator if offline or in quick-build mode.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_RAW_DIR / "messages.csv"
    
    if local_path.exists():
        print(f"Loading cached Message dataset from {local_path}...")
        return pd.read_csv(local_path)
        
    if use_real_if_available:
        try:
            print("Attempting to fetch UCI SMS Spam Collection from raw mirror...")
            tsv_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
            resp = requests.get(tsv_url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None, names=["label_str", "message"])
                df["label"] = (df["label_str"] == "spam").astype(int)
                df = df[["message", "label"]].dropna().drop_duplicates().reset_index(drop=True)
                df.to_csv(local_path, index=False)
                print(f"Successfully loaded and cached UCI SMS Spam Collection ({len(df)} rows).")
                return df
        except Exception as e:
            print(f"UCI SMS tsv download failed ({e}). Trying archive zip...")

        try:
            print("Attempting to fetch UCI SMS Spam Collection from archive.ics.uci.edu...")
            uci_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
            resp = requests.get(uci_url, timeout=8)
            if resp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for filename in z.namelist():
                        if "SMSSpamCollection" in filename:
                            with z.open(filename) as f:
                                lines = f.read().decode("utf-8", errors="ignore").splitlines()
                                rows = []
                                for line in lines:
                                    parts = line.split("\t", 1)
                                    if len(parts) == 2:
                                        label = 1 if parts[0].strip().lower() == "spam" else 0
                                        rows.append({"message": parts[1].strip(), "label": label})
                                df = pd.DataFrame(rows)
                                df.to_csv(local_path, index=False)
                                print(f"Successfully loaded and cached UCI SMS Spam Collection ({len(df)} rows).")
                                return df
        except Exception as e:
            print(f"UCI SMS download skipped/failed ({e}). Utilizing fallback synthetic generator.")
            
    print("Generating representative synthetic Message dataset (300+ samples)...")
    df = generate_synthetic_message_dataset(n_samples=360)
    df.to_csv(local_path, index=False)
    return df

def split_dataset(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict stratified split into 70% Train, 15% Val, 15% Test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # 70% Train, 30% temp
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df[label_col]
    )
    
    # Split the 30% temp evenly into 15% Val and 15% Test (50% of 30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df[label_col]
    )
    
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
