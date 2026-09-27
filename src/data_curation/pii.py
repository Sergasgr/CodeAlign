import re

SECRET_PATTERNS: dict[str, re.Pattern] = {
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "Stripe live key": re.compile(r"\b(?:sk|rk)_live_[0-9a-zA-Z]{20,}\b"),
}

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PLACEHOLDER_EMAIL = re.compile(
    r"(?i)@(?:example\.(?:com|org|net)|test\.com|localhost|users\.noreply\.github\.com)$"
)
EMAIL_TOKEN = "<EMAIL>"

def find_secrets(text: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]

def has_real_email(text: str) -> bool:
    return any(not PLACEHOLDER_EMAIL.search(m) for m in EMAIL.findall(text))

def redact_emails(text: str) -> str:
    return EMAIL.sub(lambda m: m.group(0) if PLACEHOLDER_EMAIL.search(m.group(0)) else EMAIL_TOKEN, text)