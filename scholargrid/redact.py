"""Secret redaction for logs and error reports: API keys, project tokens and database passwords."""
from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                 # Google API keys (Gemini)
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),                    # Groq
    re.compile(r"sk-or-[0-9A-Za-z\-]{20,}"),                # OpenRouter
    re.compile(r"hf_[0-9A-Za-z]{20,}"),                     # Hugging Face tokens
    re.compile(r"([?&](?:k|key)=)[A-Za-z0-9_\-]+"),         # project tokens, ?key= query params
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:/@\s]+:)[^@\s]+(?=@)"),
]


def scrub(text: str) -> str:
    for rx in _SECRET_PATTERNS:
        text = rx.sub(lambda m: (m.group(1) if m.groups() else "") + "[redacted]", text)
    return text


class SecretFilter(logging.Filter):
    """Scrubs secrets from log messages and their arguments before any handler formats them."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = scrub(str(record.msg))
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: scrub(v) if isinstance(v, str) else v for k, v in record.args.items()}
                else:
                    record.args = tuple(scrub(a) if isinstance(a, str) else a for a in record.args)
        except Exception:
            pass
        return True
