"""
feature_extractor.py
---------------------
Extracts lexical features from a URL/domain string. These features are
computed purely from the text of the URL - no internet connection,
WHOIS lookup, or DNS query required, so predictions are instant and work
completely offline.

IMPORTANT DESIGN NOTE:
The training dataset (dataset/urldata.csv) only stores the bare domain
name for each sample (e.g. "google.com", "bradesconext.digital") - not
full URLs with paths. So every feature here is calculated from the
domain string itself. This keeps training and live prediction
consistent: whatever the user types, we extract the domain the same
way, whether they typed "google.com" or "https://google.com/some/path".
"""

import re
from urllib.parse import urlparse

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|tiny\.cc|url4\.eu|su\.pr|short\.to|budurl\.com|"
    r"bit\.do|cutt\.ly|rb\.gy|shrtco\.de"
)

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "bank",
    "confirm", "signin", "password", "webscr", "suspend", "billing",
]


def get_domain(url: str) -> str:
    """Extracts just the domain (netloc) from a URL string. Works whether
    the user typed a bare domain ('google.com') or a full URL
    ('https://google.com/path')."""
    if not re.match(r'^[a-zA-Z]+://', url):
        url = 'http://' + url
    domain = urlparse(url).netloc
    # Strip a port if present, e.g. example.com:8080 -> example.com
    domain = domain.split(':')[0]
    return domain.lower()


def have_ip(domain: str) -> int:
    """1 if the domain is a raw IPv4 address instead of a name."""
    return 1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain) else 0


def have_at(original_url: str) -> int:
    """1 if '@' appears anywhere in what the user typed."""
    return 1 if "@" in original_url else 0


def domain_length(domain: str) -> int:
    """1 if the domain is long (>= 20 characters). Phishing domains tend
    to be noticeably longer than real ones (they cram in brand names +
    extra words to look convincing)."""
    return 1 if len(domain) >= 20 else 0


def count_dots(domain: str) -> int:
    """Number of '.' in the domain - lots of dots usually means lots of
    (possibly fake) subdomains, e.g. secure.paypal.login.xyz.com"""
    return domain.count('.')


def count_hyphens(domain: str) -> int:
    """Number of '-' in the domain - often used to fake brand names,
    e.g. paypal-secure-login.com"""
    return domain.count('-')


def has_digit(domain: str) -> int:
    """1 if the domain contains any digit, e.g. faceb00k.com"""
    return 1 if any(ch.isdigit() for ch in domain) else 0


def https_token_in_domain(domain: str) -> int:
    """1 if the literal word 'https' appears inside the domain name
    itself (a trick to visually look secure), e.g. https-bank-login.com"""
    return 1 if 'https' in domain else 0


def tiny_url(domain: str) -> int:
    """1 if the domain is a known URL-shortening service."""
    return 1 if SHORTENING_SERVICES.search(domain) else 0


def suspicious_keyword(domain: str) -> int:
    """1 if the domain contains a common phishing-bait keyword."""
    return 1 if any(k in domain for k in SUSPICIOUS_KEYWORDS) else 0


FEATURE_NAMES = [
    "Have_IP", "Have_At", "Domain_Length", "Count_Dots",
    "Count_Hyphens", "Has_Digit", "Https_Token_In_Domain",
    "TinyURL", "Suspicious_Keyword",
]


def extract_features(url: str) -> list:
    """Extracts all features from a raw URL/domain string typed by the
    user, returning them as a list in the exact order of FEATURE_NAMES."""
    domain = get_domain(url)

    return [
        have_ip(domain),
        have_at(url),
        domain_length(domain),
        count_dots(domain),
        count_hyphens(domain),
        has_digit(domain),
        https_token_in_domain(domain),
        tiny_url(domain),
        suspicious_keyword(domain),
    ]


if __name__ == "__main__":
    # Quick manual tests
    for test_url in [
        "google.com",
        "paypal-secure-login.verify-account.com",
        "http://192.168.1.1/login",
        "bit.ly/xyz123",
    ]:
        print(test_url, "->", dict(zip(FEATURE_NAMES, extract_features(test_url))))
