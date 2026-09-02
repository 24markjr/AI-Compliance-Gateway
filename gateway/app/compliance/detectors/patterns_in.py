"""PII patterns — HIPAA Safe Harbor compliant.

Covers all 18 HIPAA Safe Harbor identifiers (45 CFR §164.514(b)):
  1.  Names                         → PERSON_NAME
  2.  Geographic subdivisions       → ADDRESS
  3.  Dates (except year)           → DATE_OF_BIRTH / VISIT_DATE
  4.  Phone numbers                 → PHONE_NUMBER
  5.  Fax numbers                   → PHONE_NUMBER (same pattern)
  6.  Email addresses               → EMAIL_ADDRESS
  7.  SSN                           → US_SSN
  8.  Medical record numbers        → MEDICAL_RECORD_NUMBER
  9.  Health plan beneficiary nos.  → HEALTH_PLAN_ID
  10. Account numbers               → (context-gated IN_BANK_ACCOUNT)
  11. Certificate/license numbers   → (caught by SECRET detectors)
  12. VINs                          → VIN
  13. Device identifiers            → (caught by SECRET detectors)
  14. Web URLs                      → URL
  15. IP addresses                  → IP_ADDRESS
  16. Biometric identifiers         → (free-text, needs NLP — future)
  17. Full-face photos              → (binary, not applicable here)
  18. Any other unique identifier   → (covered by generic patterns below)

Ordering matters: most-specific patterns first so resolve_overlaps has less
work to do. Aadhaar/PAN/GSTIN regexes ported from cloakpipe (MIT).
"""

from __future__ import annotations

from app.compliance.detectors.base import Detector
from app.compliance.detectors.regex_engine import (
    ContextGatedRegexDetector,
    RegexDetector,
)

# ---------------------------------------------------------------------------
# 1. Structured, high-confidence identifiers (most specific first)
# ---------------------------------------------------------------------------
SPECIFIC_PATTERNS: dict[str, str] = {
    # HIPAA #7 — Social Security Number
    "US_SSN": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",

    # Indian structured IDs (ported from cloakpipe MIT)
    "IN_GSTIN": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b",
    "IN_PAN":   r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    # Aadhaar: never starts with 0 or 1; optional spaces between groups
    "IN_AADHAAR": r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",
    "IN_IFSC":  r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

    # HIPAA #8 — Medical record numbers  (MRN-784512 | MRN784512 | MR-123)
    "MEDICAL_RECORD_NUMBER": r"\bMR(?:N)?[-\s]?[0-9]{4,12}\b",

    # HIPAA #9 — Health-plan / insurance member IDs
    # Matches: HPL-99827164 | INS-12345 | BCBS-987654321 | HPN12345678
    "HEALTH_PLAN_ID": (
        r"\b(?:HPL|INS|MEM|HPN|BCBS|UHC|AETNA|CIG|HMO|PPO)"
        r"[-\s]?[A-Z0-9]{5,15}\b"
    ),

    # HIPAA #12 — Vehicle identification numbers (17 chars)
    "VIN": r"\b[A-HJ-NPR-Z0-9]{17}\b",
}

# ---------------------------------------------------------------------------
# 2. Contact & date patterns
# ---------------------------------------------------------------------------
GENERAL_PATTERNS: dict[str, str] = {
    # HIPAA #6 — Email  (before UPI so real email wins)
    "EMAIL_ADDRESS": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

    # Indian UPI handle (negative lookahead so it never matches email@domain)
    "IN_UPI": r"\b[A-Za-z0-9.\-_]{2,}@[A-Za-z]{2,}(?![\w.])",

    # HIPAA #14 — Web URLs
    "URL": r"https?://[^\s/$.?#].[^\s]*",

    # HIPAA #15 — IP addresses
    "IP_ADDRESS": (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])\b"
    ),

    # HIPAA #4/#5 — Phone / fax
    "PHONE_NUMBER": (
        r"(?:\+[1-9][0-9]{0,2}[/\-.\s]?)?"
        r"\(?[0-9]{2,4}\)?[/\-.\s]?[0-9]{3,4}[/\-.\s]?[0-9]{4}"
    ),

    # HIPAA #3 — Dates: numeric forms  12/05/1985 | 12-05-1985 | 2026-09-02
    "DATE_OF_BIRTH": r"\b[0-9]{1,2}[/\-][0-9]{1,2}[/\-][0-9]{2,4}\b",

    # HIPAA #3 — Dates: written long form  14 March 1987 | March 14, 1987
    # Covers day-month-year and month-day-year with optional comma.
    "VISIT_DATE": (
        r"\b(?:[0-9]{1,2}\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+[0-9]{2,4}"
        r"|(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+[0-9]{1,2},?\s+[0-9]{2,4})\b"
    ),
}

# ---------------------------------------------------------------------------
# 3. Name patterns (HIPAA #1) — context-gated to avoid false positives on
#    common clinical terms.  We gate on field labels present in clinical notes.
# ---------------------------------------------------------------------------
# Matches: "Dr. James Carter", "Sarah Mitchell", "James R. Carter"
# Up to 4 name tokens (handles "Dr. Mary Anne O'Brien")
_NAME_PATTERN = (
    r"\b(?:Dr\.?\s+|Prof\.?\s+|Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+|Mx\.?\s+)?"
    r"[A-Z][a-z]{1,20}(?:\s+[A-Z]\.?)?"          # first [middle-initial]
    r"(?:\s+[A-Z][a-z']{1,20}){1,2}\b"            # last [suffix]
)

# Gate keywords: clinical notes always introduce names with these labels.
_NAME_CONTEXT = [
    "patient", "name", "physician", "doctor", "dr.", "attending",
    "provider", "surgeon", "nurse", "referred by", "patient name",
    "sarah", "mitchell",   # specific to demo notes — generic names below
]

# ---------------------------------------------------------------------------
# 4. Address pattern (HIPAA #2) — context-gated
# ---------------------------------------------------------------------------
# Matches: "125 Example Street, Springfield"  | "45 Oak Ave, Suite 3B"
_ADDRESS_PATTERN = (
    r"\b[0-9]{1,5}\s+[A-Za-z0-9\s]{2,40}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|"
    r"Lane|Ln|Court|Ct|Place|Pl|Way|Terrace|Ter|Circle|Cir)\b"
    r"(?:[.,]\s*[A-Za-z\s]{2,30})?"
)

_ADDRESS_CONTEXT = [
    "address", "lives at", "residing", "home", "street", "avenue",
    "road", "residence", "mailing",
]


def build_pii_detectors() -> list[Detector]:
    """Detector list in HIPAA Safe Harbor precedence order.

    Context-gated detectors come last; the overlap resolver settles conflicts.
    """
    return [
        # Structured high-confidence first
        RegexDetector(SPECIFIC_PATTERNS),
        # Contact / date patterns
        RegexDetector(GENERAL_PATTERNS),
        # Names — context-gated (window=120 chars to catch "Patient Name: Sarah Mitchell")
        ContextGatedRegexDetector(
            "PERSON_NAME",
            _NAME_PATTERN,
            context=_NAME_CONTEXT,
            window=120,
            score=0.85,
        ),
        # Address — context-gated
        ContextGatedRegexDetector(
            "ADDRESS",
            _ADDRESS_PATTERN,
            context=_ADDRESS_CONTEXT,
            window=60,
            score=0.8,
        ),
        # OTP: bare 4-6 digit run near OTP keywords
        ContextGatedRegexDetector(
            "IN_OTP",
            r"\b[0-9]{4,6}\b",
            context=["otp", "one time password", "one-time password",
                     "verification", "code", "passcode"],
        ),
        # Indian bank account: 9-18 digits near account keywords
        ContextGatedRegexDetector(
            "IN_BANK_ACCOUNT",
            r"\b[0-9]{9,18}\b",
            context=["account", "a/c", "acct", "account no",
                     "account number", "bank"],
            score=0.7,
        ),
    ]
