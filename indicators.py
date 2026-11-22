import re

# ======================================================================
# Constants
# ======================================================================
SEVERITY_LEVELS = ["Low", "Medium", "High"]          # weakest -> strongest
SEVERITY_POINTS = {"Low": 1, "Medium": 2, "High": 3}  # used for the overall level

# Category names (defined once so the UI and the tests use the same text)
CAT_URGENCY = "Urgency and Pressure"
CAT_LINKS = "Suspicious Links"
CAT_CREDENTIALS = "OTP and Credential Requests"
CAT_FINANCIAL = "Financial Requests"
CAT_PRIZE = "Prize and Reward Language"
CAT_THREATS = "Threats and Account Suspension"

# Display order for ties in severity
CATEGORY_ORDER = [
    CAT_CREDENTIALS, CAT_THREATS, CAT_LINKS,
    CAT_URGENCY, CAT_FINANCIAL, CAT_PRIZE,
]

METHOD_NOTE = (
    "Rule-based keyword and pattern checks. This is NOT a machine learning "
    "prediction and is separate from the model's score."
)

SUMMARIES = {
    "None": "No rule-based scam indicators were found. This does not prove the message is safe.",
    "Low": "Only weak rule-based indicators were found. Stay careful, especially if the message was unexpected.",
    "Medium": "Some rule-based scam indicators were found. Treat the message with caution and verify it through an official channel.",
    "High": "Strong rule-based scam indicators were found (several categories or high severity). Do not click links or share any information.",
}


# ======================================================================
# Rules for the five keyword-based categories
# ======================================================================
# Each pattern is written as:  _p(severity, regex, ignore_if_negated)
#   - regex is case-insensitive.
#   - ignore_if_negated=True means: skip the match when it sits in a warning
#     such as "Do not share your OTP" (real banks send those warnings).

def _p(severity, regex, ignore_if_negated=False):
    return (severity, re.compile(regex, re.IGNORECASE), ignore_if_negated)


# Words that name a secret credential. Used by the credential rules below.
# "pin(?!\s*code)" skips "PIN code", which is just a postal code in India.
CRED_WORDS = (
    r"(?:otp|one[- ]time\s+(?:password|pin|code)|mpin|upi\s+pin|pin(?!\s*code)"
    r"|password|passcode|cvv|cvc|card\s+(?:number|details)"
    r"|login\s+(?:details|credentials)|credentials|security\s+code"
    r"|verification\s+code|net\s*banking\s+(?:id|password))"
)

# "share/send/enter ... OTP/password/PIN"  (a request for a credential)
CRED_REQUEST = (
    r"\b(?:share|send|give|tell|provide|enter|submit|reply(?:\s+with)?|forward"
    r"|reveal|disclose|read\s+out|confirm|verify|type)\b[^.!?\n]{0,40}?\b"
    + CRED_WORDS + r"\b"
)
# Just the credential word on its own (a weak signal)
CRED_MENTION = r"\b" + CRED_WORDS + r"\b"


RULES = [
    # ------------------------------------------------------------------
    {
        "category": CAT_URGENCY,
        "explanation": (
            "Scammers create time pressure so you act before you think. "
            "Genuine organisations rarely demand immediate action by text message."
        ),
        "patterns": [
            _p("High", r"\bact\s+(?:now|fast|immediately|quickly)\b"),
            _p("High", r"\b(?:immediate|urgent)\s+action\b"),
            _p("High", r"\baction\s+required\b"),
            _p("High", r"\bfinal\s+(?:notice|warning|reminder|call)\b"),
            _p("High", r"\blast\s+(?:chance|warning|reminder)\b"),
            _p("Medium", r"\burgent(?:ly)?\b"),
            _p("Medium", r"\bimmediately\b"),
            _p("Medium", r"\bright\s+away\b"),
            _p("Medium", r"\basap\b"),
            _p("Medium", r"\bwithin\s+(?:the\s+next\s+)?(?:\d+|one|two|three|a|an)\s*"
                         r"(?:hours?|hrs?|minutes?|mins?|days?)\b"),
            _p("Medium", r"\b(?:today\s+only|limited\s+time|offer\s+(?:ends|expires)"
                         r"|expires?\s+(?:today|tonight|soon))\b"),
            _p("Medium", r"\b(?:click|call|reply|respond|claim|pay|verify|update|log\s?in"
                         r"|confirm|text|txt|order|apply|register|download|install)"
                         r"\s+(?:here\s+)?(?:now|immediately|today|asap|urgently)\b"),
            _p("Low", r"\bhurry\b"),
            _p("Low", r"\bdon['’]?t\s+(?:miss|delay|wait|ignore)\b"),
            _p("Low", r"\bdo\s+not\s+(?:ignore|delay|miss)\b"),
            _p("Low", r"\bas\s+soon\s+as\s+possible\b"),
            _p("Low", r"\bbefore\s+it['’]?s\s+too\s+late\b"),
        ],
    },
    # ------------------------------------------------------------------
    {
        "category": CAT_THREATS,
        "explanation": (
            "Threats such as blocked accounts, legal action or penalties are used "
            "to frighten people into obeying. Real banks and agencies do not "
            "usually threaten you this way by SMS or email."
        ),
        "patterns": [
            _p("High", r"\b(?:account|card|sim|wallet|access|number)\s+"
                       r"(?:has\s+been\s+|have\s+been\s+|will\s+be\s+|is\s+|was\s+|been\s+|gets?\s+)?"
                       r"(?:suspended|blocked|locked|closed|deactivated|frozen|disabled"
                       r"|terminated|disconnected|barred|restricted)\b"),
            _p("High", r"\baccount\s+(?:closure|suspension|termination|deactivation)\b"),
            _p("High", r"\blegal\s+(?:action|proceedings|notice|case|consequences)\b"),
            _p("High", r"\b(?:arrest(?:ed)?|warrant|lawsuit|prosecut(?:ed|ion)|jail|imprisonment)\b"),
            _p("Medium", r"\b(?:will|shall|may)\s+(?:be\s+)?(?:suspended|blocked|terminated"
                         r"|deactivated|disconnected|barred|closed|frozen)\b"),
            _p("Medium", r"\bavoid\s+(?:account\s+)?(?:suspension|blocking|closure|penalty"
                         r"|penalties|disconnection|deactivation|legal)\b"),
            _p("Medium", r"\bunauthori[sz]ed\s+(?:access|transaction|login|activity|attempt|payment|use)\b"),
            _p("Medium", r"\bsuspicious\s+(?:activity|login|transaction|attempt|sign[- ]?in)\b"),
            _p("Medium", r"\b(?:security|fraud)\s+(?:alert|warning|breach|notice)\b"),
            _p("Medium", r"\b(?:compromised|hacked|breached?)\b"),
            _p("Medium", r"\bpenalt(?:y|ies)\b"),
            _p("Medium", r"\bfailure\s+to\s+(?:comply|respond|verify|update|pay)\b"),
        ],
    },
    # ------------------------------------------------------------------
    {
        "category": CAT_CREDENTIALS,
        "explanation": (
            "Legitimate organisations never ask you to share OTPs, PINs, passwords "
            "or CVV codes. Anyone who has them can take over your account."
        ),
        "patterns": [
            # A request to hand over a credential. Ignored in warnings such as
            # "Do not share your OTP with anyone".
            _p("High", CRED_REQUEST, ignore_if_negated=True),
            _p("Medium", r"\b(?:verify|confirm|validate|update|re-?activate|re-?verify)\s+"
                         r"(?:your\s+)?(?:account|identity|details|information|bank|card"
                         r"|kyc|profile|login|credentials|password)\b"),
            _p("Low", r"\b(?:log\s?in|sign\s?in)\s+(?:to|here|now|below|using|via)\b"),
            _p("Low", CRED_MENTION),
        ],
    },
    # ------------------------------------------------------------------
    {
        "category": CAT_FINANCIAL,
        "explanation": (
            "Requests involving bank details, cards, fees or transfers are how many "
            "scams collect money or financial data. Check any such request through "
            "the organisation's official app or phone number."
        ),
        "patterns": [
            _p("High", r"\b(?:bank\s+)?account\s+(?:number|details)\b"),
            _p("High", r"\b(?:processing|registration|clearance|customs|delivery|release"
                       r"|handling|admin|activation|redelivery|re-delivery)\s+"
                       r"(?:fee|fees|charge|charges)\b"),
            _p("High", r"\b(?:western\s+union|moneygram|wire\s+transfer)\b"),
            _p("High", r"\badvance\s+(?:fee|payment)\b"),
            _p("Medium", r"\b(?:credit|debit)\s+card\b"),
            _p("Medium", r"\bnet\s*banking\b"),
            _p("Medium", r"\bbank\s+account\b"),
            _p("Medium", r"\bupi\b"),
            _p("Medium", r"\bkyc\b"),
            _p("Medium", r"\baadhaar\b"),
            _p("Medium", r"\bpan\s+(?:card|number)\b"),
            _p("Medium", r"\brefund\b"),
            _p("Medium", r"\bcashback\b"),
            _p("Medium", r"\bgift\s+cards?\b"),
            _p("Medium", r"\b(?:bitcoin|crypto(?:currency)?|usdt)\b"),
            _p("Medium", r"\bpayment\s+(?:failed|pending|due|required|declined)\b"),
            _p("Medium", r"\bunpaid\b"),
            _p("Medium", r"\b(?:send|transfer|wire|deposit|pay)\b[^.!?\n]{0,30}?\b"
                         r"(?:money|funds|amount|fee|fees|payment|charges)\b"),
            # An amount of money on its own is common in normal messages too
            _p("Low", r"(?:[£$€₹]\s?\d[\d,]*(?:\.\d+)?"
                      r"|\b(?:rs\.?|inr|usd|gbp|eur)\s?\d[\d,]*(?:\.\d+)?"
                      r"|\b\d[\d,]*(?:\.\d+)?\s?(?:rs|inr|usd|gbp|dollars|rupees|pounds)\b)"),
        ],
    },
    # ------------------------------------------------------------------
    {
        "category": CAT_PRIZE,
        "explanation": (
            "Unexpected prizes, lotteries and rewards are classic bait. You cannot "
            "win a contest you never entered, and 'claiming' a prize usually costs "
            "money or personal data."
        ),
        "patterns": [
            _p("High", r"\b(?:you|u)(?:['’]ve|\s+have|\s+are)?\s+(?:just\s+)?won\b"),
            _p("High", r"\b(?:lottery|lucky\s+draw|jackpot|sweepstakes?)\b"),
            _p("High", r"\bclaim\s+(?:your|ur|the|a)\s+(?:free\s+)?(?:prize|reward|gift"
                       r"|winnings?|cash|bonus|voucher|refund)\b"),
            _p("High", r"\b(?:cash\s+prize|prize\s+(?:money|draw)|grand\s+prize)\b"),
            _p("Medium", r"\b(?:congratulations|congrats)\b"),
            _p("Medium", r"\bwinner\b"),
            _p("Medium", r"\bprizes?\b"),
            _p("Medium", r"\b(?:you|u)(?:['’]ve|\s+have|\s+are)?\s+(?:been\s+)?"
                         r"(?:selected|chosen|awarded)\b"),
            _p("Medium", r"\bfree\s+(?:entry|gift|prize|cash|ringtone|voucher|iphone|ipad"
                         r"|laptop|trip|holiday|vacation|cruise|tickets?)\b"),
            _p("Medium", r"\b(?:giveaway|raffle|gift\s+voucher)\b"),
            _p("Medium", r"\blucky\s+(?:winner|customer|user|number)\b"),
            _p("Low", r"\bclaim\b"),
            _p("Low", r"\brewards?\b"),
            _p("Low", r"\bbonus\b"),
            _p("Low", r"\bguaranteed\b"),
            _p("Low", r"\bexclusive\s+offer\b"),
            _p("Low", r"\bfree\s+(?:msg|sms|text|calls?|data|minutes|mins|recharge|membership|trial)\b"),
        ],
    },
]

# Words that turn a match into a warning ("do not share your OTP").
_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|never|not|no\s+one|nobody|won['’]?t"
    r"|should\s+not|shouldn['’]?t|will\s+never)\b",
    re.IGNORECASE,
)
# Punctuation that ends a clause, so a negation in an earlier clause is ignored
_CLAUSE_BREAK_RE = re.compile(r"[.!?;:,\n]")


# ======================================================================
# Helper functions (small and reusable)
# ======================================================================
def _is_negated(text, start):
    """True if the words just before `start` (in the same clause) contain a
    negation such as 'do not' or 'never'. Example: 'Never share your OTP'."""
    clause = _CLAUSE_BREAK_RE.split(text[:start])[-1]
    return bool(_NEGATION_RE.search(clause[-40:]))


def _merge_spans(spans):
    """Merge overlapping (start, end) ranges into the smallest set of ranges.
    Example: [(0, 5), (3, 9)] -> [(0, 9)]."""
    merged = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _raise_one_level(severity):
    """Low -> Medium -> High (High stays High)."""
    index = min(SEVERITY_LEVELS.index(severity) + 1, len(SEVERITY_LEVELS) - 1)
    return SEVERITY_LEVELS[index]


def _unique_evidence(text, spans):
    """The matched pieces of text, without repeats (case-insensitive)."""
    seen, pieces = set(), []
    for start, end in spans:
        piece = text[start:end].strip()
        if piece and piece.lower() not in seen:
            seen.add(piece.lower())
            pieces.append(piece)
    return pieces


def _make_finding(text, category, severity, spans, explanation):
    """Build one finding dictionary (the structure the UI will read)."""
    if severity == "Low":
        explanation = "Weak signal on its own. " + explanation
    return {
        "category": category,
        "severity": severity,
        "evidence": _unique_evidence(text, spans),
        "spans": spans,
        "explanation": explanation,
    }


def _check_pattern_rule(text, rule):
    """Run every pattern of one category. Returns a finding or None."""
    matches = []  # (start, end, severity)
    for severity, regex, ignore_if_negated in rule["patterns"]:
        for m in regex.finditer(text):
            if ignore_if_negated and _is_negated(text, m.start()):
                continue  # e.g. "Do not share your OTP" is a warning, not a request
            matches.append((m.start(), m.end(), severity))

    if not matches:
        return None

    # Category severity = the strongest single match...
    severity = max((m[2] for m in matches), key=SEVERITY_LEVELS.index)
    spans = _merge_spans([(start, end) for start, end, _ in matches])
    # ...raised one level when there are 3+ separate pieces of evidence.
    if len(spans) >= 3:
        severity = _raise_one_level(severity)

    return _make_finding(text, rule["category"], severity, spans, rule["explanation"])


# ======================================================================
# Link analysis (needs more than keywords, so it has its own code)
# ======================================================================
# Domain endings we recognise when a link has no "http://" in front of it
BARE_DOMAIN_TLDS = [
    "com", "net", "org", "info", "biz", "xyz", "top", "club", "online", "site",
    "click", "link", "live", "icu", "vip", "work", "tk", "ml", "ga", "cf", "gq",
    "co", "in", "me", "ly", "gl", "cc", "ws", "us", "uk",
]

_URL_CHARS = r"""[^\s<>"']+"""
URL_RE = re.compile(
    r"https?://" + _URL_CHARS                     # http://... or https://...
    + r"|www\." + _URL_CHARS                      # www.example.com/...
    # bare domains such as bit.ly/abc. The lookbehind stops e-mail addresses
    # (john@example.com) from being treated as links.
    + r"|(?<![@\w.\-])(?:[a-z0-9\-]+\.)+(?:" + "|".join(BARE_DOMAIN_TLDS) + r")\b(?:/" + _URL_CHARS + r")?",
    re.IGNORECASE,
)
TRAILING_PUNCTUATION = ".,;:!?)]}\"'"

# Link shorteners hide the real destination
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy",
    "shorturl.at", "tiny.cc", "rebrand.ly", "buff.ly", "bit.do", "s.id", "lnkd.in",
    "t.ly", "v.gd", "clck.ru", "shorte.st", "adf.ly",
}
# Domain endings that are cheap and often used by throwaway scam sites
SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "click", "icu", "vip", "work", "tk", "ml", "ga", "cf",
    "gq", "link", "online", "site", "live",
}
# Words that phishing links often contain
LINK_WORDS = [
    "login", "signin", "verify", "secure", "update", "account", "bank", "kyc",
    "confirm", "wallet", "password", "otp", "refund", "claim", "prize", "payment",
]


def _get_host(url):
    """Split a (lower-case) link into (host name, has '@' trick)."""
    without_scheme = re.sub(r"^https?://", "", url)
    authority = re.split(r"[/?#]", without_scheme, maxsplit=1)[0]
    has_at = "@" in authority
    host = authority.split("@")[-1].split(":")[0]
    return host, has_at


def _link_red_flags(url):
    """Return a list of (severity, reason) for one link."""
    flags = []
    lower = url.lower()
    host, has_at = _get_host(lower)
    tld = host.rsplit(".", 1)[-1]

    if lower.startswith("http://"):
        flags.append(("Low", "it uses plain http instead of secure https"))
    if host.removeprefix("www.") in SHORTENERS:
        flags.append(("Medium", "it is a shortened link that hides the real destination"))
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        flags.append(("High", "it points to a raw IP address instead of a normal website name"))
    if has_at:
        flags.append(("High", "it contains an '@' sign, a trick used to disguise the real destination"))
    if tld in SUSPICIOUS_TLDS:
        flags.append(("Medium", f"its '.{tld}' ending is common on throwaway scam sites"))
    if "xn--" in host:
        flags.append(("Medium", "its domain uses look-alike (punycode) characters"))
    if host.count("-") >= 2:
        flags.append(("Medium", "its domain has many hyphens, typical of fake look-alike sites"))
    if len(host.split(".")) >= 5:
        flags.append(("Medium", "it has many sub-domains, a trick used to imitate real brands"))

    found_words = [w for w in LINK_WORDS if w in lower]
    if found_words:
        flags.append(("Low", f"it contains words phishing links often use ({', '.join(found_words[:3])})"))
    return flags


def _check_links(text):
    """Find links in the message and judge them. Returns a finding or None."""
    spans, reasons = [], []
    for match in URL_RE.finditer(text):
        url = match.group().rstrip(TRAILING_PUNCTUATION)
        if not url:
            continue
        spans.append((match.start(), match.start() + len(url)))
        reasons.extend(_link_red_flags(url))

    if not spans:
        return None

    # A link alone is at least Low; red flags can raise it.
    severity = "Low"
    for level, _ in reasons:
        if SEVERITY_LEVELS.index(level) > SEVERITY_LEVELS.index(severity):
            severity = level

    sentences = []
    for _, reason in reasons:
        if reason not in sentences:
            sentences.append(reason)

    explanation = ("Links in unexpected messages are a common way to send people "
                   "to fake websites.")
    if sentences:
        explanation += " Red flags found: " + "; ".join(sentences) + "."
    else:
        explanation += (" No specific red flag was found in the link itself, but do "
                        "not open links you did not expect.")

    return _make_finding(text, CAT_LINKS, severity, spans, explanation)


# ======================================================================
# Overall evidence level + the main function
# ======================================================================
def _overall_level(findings):
    """Combine findings into None / Low / Medium / High (and a numeric score).
    Low=1, Medium=2, High=3 points per category.
      High   : score >= 6, or two or more High findings
      Medium : score 3-5
      Low    : score 1-2"""
    if not findings:
        return "None", 0
    score = sum(SEVERITY_POINTS[f["severity"]] for f in findings)
    high_count = sum(1 for f in findings if f["severity"] == "High")
    if score >= 6 or high_count >= 2:
        return "High", score
    if score >= 3:
        return "Medium", score
    return "Low", score


def analyze(text):
    """Analyze one message and return all detected indicators.

    Returns a dictionary (see the top of this file for the layout).
    Never raises on odd input: empty or non-text input gives 'None'."""
    if not isinstance(text, str):
        text = ""

    findings = []
    for rule in RULES:
        finding = _check_pattern_rule(text, rule)
        if finding:
            findings.append(finding)
    link_finding = _check_links(text)
    if link_finding:
        findings.append(link_finding)

    # Strongest first; ties follow CATEGORY_ORDER
    findings.sort(key=lambda f: (-SEVERITY_LEVELS.index(f["severity"]),
                                 CATEGORY_ORDER.index(f["category"])))

    level, score = _overall_level(findings)
    all_spans = [span for f in findings for span in f["spans"]]

    return {
        "method": METHOD_NOTE,
        "evidence_level": level,
        "score": score,
        "summary": SUMMARIES[level],
        "findings": findings,
        "categories_found": [f["category"] for f in findings],
        "highlight_spans": _merge_spans(all_spans),
    }


# ======================================================================
# Self-test: run with  python indicators.py
# ======================================================================
def _run_self_tests():
    # "has": category -> expected severity (None = any severity is fine)
    tests = [
        {
            "name": "Prize scam with shortened link",
            "text": "Congratulations! You have won a £1000 cash prize. Claim now at "
                    "bit.ly/win123 or call 09061701461 within 24 hours.",
            "level": "High",
            "has": {CAT_PRIZE: "High", CAT_LINKS: "Medium", CAT_URGENCY: None},
        },
        {
            "name": "Bank phishing (threat + KYC + suspicious link)",
            "text": "ALERT: Your SBI account has been blocked. Update your KYC immediately "
                    "at http://sbi-kyc-update.xyz or your account will be suspended within 24 hours.",
            "level": "High",
            "has": {CAT_THREATS: "High", CAT_LINKS: "Medium", CAT_URGENCY: None,
                    CAT_FINANCIAL: None, CAT_CREDENTIALS: None},
        },
        {
            "name": "OTP request",
            "text": "Dear customer, to avoid account closure please share your OTP "
                    "482913 with our agent immediately.",
            "level": "High",
            "has": {CAT_CREDENTIALS: "High", CAT_THREATS: None, CAT_URGENCY: None},
        },
        {
            "name": "Customs-fee scam with raw IP link",
            "text": "Your parcel is held. Pay the customs fee at http://192.168.10.5/track now.",
            "level": "High",
            "has": {CAT_LINKS: "High", CAT_FINANCIAL: "High"},
        },
        {
            "name": "Classic SMS-spam style prize message",
            "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize "
                    "Jackpot! Txt the word: CLAIM to No: 81010",
            "level": "High",
            "has": {CAT_PRIZE: "High", CAT_URGENCY: None},
        },
        {
            "name": "Legitimate OTP message (warning must NOT count as a request)",
            "text": "123456 is your OTP for login to ExampleBank. Do not share it with "
                    "anyone, including bank staff.",
            "level": "Low",
            "has": {CAT_CREDENTIALS: "Low"},
        },
        {
            "name": "Normal friendly message",
            "text": "Hey, are you free for dinner tonight? Let me know what time works for you.",
            "level": "None",
            "has": {},
        },
        {
            "name": "E-mail address is not a link",
            "text": "Please contact me at john.smith@example.com tomorrow.",
            "level": "None",
            "has": {},
            "lacks": [CAT_LINKS],
        },
        {"name": "Empty message", "text": "", "level": "None", "has": {}},
        {"name": "Whitespace only", "text": "   \n  ", "level": "None", "has": {}},
    ]

    print("=" * 66)
    print("INDICATOR SELF-TEST")
    print("=" * 66)
    failures = 0
    for test in tests:
        text = test["text"]
        result = analyze(text)
        found = {f["category"]: f["severity"] for f in result["findings"]}
        problems = []

        if result["evidence_level"] != test["level"]:
            problems.append(f"level is {result['evidence_level']}, expected {test['level']}")
        for category, severity in test.get("has", {}).items():
            if category not in found:
                problems.append(f"missing category: {category}")
            elif severity is not None and found[category] != severity:
                problems.append(f"{category}: severity {found[category]}, expected {severity}")
        for category in test.get("lacks", []):
            if category in found:
                problems.append(f"should NOT contain: {category}")
        # Sanity checks: evidence really comes from the text, spans are valid
        for finding in result["findings"]:
            for piece in finding["evidence"]:
                if piece.lower() not in text.lower():
                    problems.append(f"evidence not found in text: {piece!r}")
        for start, end in result["highlight_spans"]:
            if not (0 <= start < end <= len(text)):
                problems.append(f"bad highlight span: {(start, end)}")

        status = "PASS" if not problems else "FAIL"
        print(f"[{status}] {test['name']}  ->  level: {result['evidence_level']}")
        for problem in problems:
            print(f"        - {problem}")
        failures += bool(problems)

    print("-" * 66)
    print("All tests passed." if not failures else f"{failures} test(s) FAILED.")

    # Demo: what the UI will receive for one message
    demo = analyze(tests[1]["text"])
    print("\nDEMO OUTPUT")
    print(f"Message : {tests[1]['text']}")
    print(f"Level   : {demo['evidence_level']}  (score {demo['score']})")
    print(f"Summary : {demo['summary']}")
    for finding in demo["findings"]:
        print(f"\n  [{finding['severity']}] {finding['category']}")
        print(f"    Evidence   : {finding['evidence']}")
        print(f"    Why        : {finding['explanation']}")
    print(f"\nNote: {demo['method']}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()