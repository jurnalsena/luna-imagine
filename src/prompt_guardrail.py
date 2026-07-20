"""Custom prompt guardrail to block NSFW / pornographic generation requests."""

import re
import unicodedata

# Configuration: Set to False to disable content guardrail entirely
ENABLE_GUARDRAIL = True

# Whole-word terms (English + Indonesian). Kept in a set for fast lookup.
_BLOCKED_TERMS = {
    # Explicit / pornography
    "porn", "porno", "pornography", "pornographic", "xxx", "nsfw", "hentai", "ecchi",
    "bokep", "jav", "xvideo", "xvideos", "redtube", "youporn", "onlyfans",
    # Nudity / exposure
    "nude", "nudes", "nudity", "naked", "topless", "bottomless", "bare breasts",
    "full frontal", "exposed breasts", "exposed genitals", "no clothes", "without clothes",
    "telanjang", "bugil", "payudara", "vagina", "penis", "testicle", "testicles",
    "genital", "genitals", "areola", "nipple", "nipples", "labia", "clitoris",
    # Sexual acts
    "sex", "sexual", "intercourse", "blowjob", "handjob", "fellatio", "cunnilingus",
    "anal sex", "oral sex", "gangbang", "threesome", "orgy", "masturbat", "masturbation",
    "ejaculat", "cumshot", "creampie", "deepthroat", "doggy style", "missionary position",
    "sodomy", "bestiality", "zoophilia", "rape", "raping", "molest", "molestation",
    "pemesek", "ngewe", "ngentot", "bersetubuh", "seks", "seksual",
    # Body / fetish (commonly used in NSFW prompts)
    "fetish", "bdsm", "bondage", "dominatrix", "stripper", "strip club", "lap dance",
    "upskirt", "downblouse", "cameltoe", "underboob", "sideboob", "wardrobe malfunction",
    "see through", "see-through", "sheer lingerie", "micro bikini", "g-string",
    # Slurs / minors (zero tolerance)
    "loli", "lolicon", "shota", "shotacon", "child porn", "cp ", "underage",
    "minor nude", "teen nude", "schoolgirl nude",
}

# Regex patterns for phrases, evasions, and compound NSFW intent.
_BLOCKED_PATTERNS = [
    r"\bnsfw\b",
    r"\bxxx+\b",
    r"\b(?:p0rn|pr0n|nud3|n@ked|s3x|s€x)\b",
    r"\b(?:no|without|without any|w\/o)\s+(?:clothes|clothing|outfit|panties|bra|underwear)\b",
    r"\b(?:remove|removing|take off|taking off)\s+(?:clothes|clothing|bra|panties|underwear|top|shirt)\b",
    r"\b(?:spread(?:ing)?|open(?:ing)?)\s+(?:legs|thighs)\b",
    r"\b(?:touch(?:ing)?|grab(?:bing)?|grop(?:ing|e))\s+(?:breasts?|boobs?|butt|genitals?)\b",
    r"\b(?:nude|naked|nsfw)\s+(?:girl|woman|women|boy|man|teen|model|photo|pic|image|selfie)\b",
    r"\b(?:erotic|sensual|provocative)\s+(?:photo|pose|picture|scene|art)\b.*\b(?:nude|naked|explicit)\b",
    r"\b(?:explicit|graphic)\s+(?:sex|sexual|nude|adult)\b",
    r"\b(?:adult|18\+)\s+(?:content|video|scene|film)\b",
    r"\b(?:onlyfans|playboy|penthouse)\s+(?:style|model|photo)\b",
    r"\b(?:bokep|film\s*dewasa|video\s*dewasa)\b",
    r"\b(?:telanjang|bugil)\s+(?:wanita|perempuan|cewek|gadis|pria|cowok)\b",
]

_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i",
})

_GUARDRAIL_MESSAGE = (
    "This tool does not allow pornographic, sexually explicit, or NSFW requests.\n"
    "Please revise your prompt and try again."
)


def _normalize(text: str) -> str:
    """Lowercase, strip accents, and reduce common obfuscation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.translate(_LEET_MAP)
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_blocked_term(text: str) -> str | None:
    """Return the first blocked whole-word term found, if any."""
    padded = f" {text} "
    for term in sorted(_BLOCKED_TERMS, key=len, reverse=True):
        needle = f" {term} "
        if needle in padded:
            return term
        if term.isalpha() and re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return None


def _find_blocked_pattern(text: str) -> str | None:
    """Return the first blocked regex pattern that matches, if any."""
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def check_prompt_safety(prompt: str) -> tuple[bool, str | None]:
    """
    Check whether a prompt is safe to send to the generator.

    Returns:
        (True, None) if safe or guardrail is disabled.
        (False, reason) if blocked.
    """
    if not ENABLE_GUARDRAIL:
        return True, None
    
    normalized = _normalize(prompt)
    if not normalized:
        return True, None

    blocked_term = _find_blocked_term(normalized)
    if blocked_term:
        return False, f"Blocked term detected: '{blocked_term}'"

    blocked_pattern = _find_blocked_pattern(normalized)
    if blocked_pattern:
        return False, "Blocked phrase pattern detected."

    return True, None


def enforce_prompt_safety(prompt: str, *, field_name: str = "Prompt") -> None:
    """
    Validate prompt safety and raise ValueError with a user-facing message if blocked.

    Use this from generation handlers before calling the backend.
    """
    if not ENABLE_GUARDRAIL:
        return
    
    is_safe, reason = check_prompt_safety(prompt)
    if not is_safe:
        detail = reason or "Unsafe content detected."
        raise ValueError(f"{field_name} blocked by content guardrail.\n\n{detail}\n\n{_GUARDRAIL_MESSAGE}") from None
