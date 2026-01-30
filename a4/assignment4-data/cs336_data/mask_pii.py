import re

def mask_emails(text: str):
    email_pattern = r'[\w.]+@[\w.]+\.[a-z]+'
    replacement = "|||EMAIL_ADDRESS|||"
    mask_text, count = re.subn(email_pattern, replacement, text)

    return mask_text, count

def mask_phone_numbers(text: str):
    phone_pattern = r'(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    replacement = "|||PHONE_NUMBER|||"
    mask_text, count = re.subn(phone_pattern, replacement, text)

    return mask_text, count

def mask_ipv4(text: str):
    chunk = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    ipv4_pattern = r"\b(?:" + chunk + r"\.){3}" + chunk + r"\b"
    replacement = "|||IP_ADDRESS|||"
    mask_text, count = re.subn(ipv4_pattern, replacement, text)

    return mask_text, count