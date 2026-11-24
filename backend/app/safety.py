"""A simple check for text that may signal an urgent risk of self-harm.

This is a phrase list, not a model. It misses many cases (false negatives) and
sometimes fires on harmless text (false positives). It is a safety net only.
It never stores or logs the text.
"""

import re

_PHRASES = [
    r"kill(?:ing)?\s+myself",
    r"end(?:ing)?\s+my\s+(?:own\s+)?life",
    r"take\s+my\s+own\s+life",
    r"want(?:ed)?\s+to\s+die",
    r"wish\s+i\s+(?:was|were)\s+dead",
    r"better\s+off\s+dead",
    r"don'?t\s+want\s+to\s+(?:live|be\s+alive|exist)\b",
    r"no\s+reason\s+to\s+live",
    r"suicid(?:e|al)",
    r"(?:hurt|hurting|harm|harming|cut|cutting)\s+myself",
    r"self[-\s]?harm",
    r"end\s+it\s+all",
]
_PATTERN = re.compile("|".join(f"(?:{p})" for p in _PHRASES), re.IGNORECASE)


def check_urgent(text):
    """True if the text contains wording that may signal an urgent risk."""
    if not isinstance(text, str):
        return False
    return bool(_PATTERN.search(text.replace("\u2019", "'")))