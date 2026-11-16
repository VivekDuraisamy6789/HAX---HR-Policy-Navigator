"""Safety layer: request screening, prompt-injection handling, PII redaction.

Design note: the original keyword list ("salary", "fire", ...) refused legitimate questions
("What is the fire drill procedure?", "How is salary reviewed?"). Requests are now screened
by *intent patterns* (a directive to take a consequential action, or a request for another
person's private data) and the planner LLM provides a second opinion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

REFUSAL_TEXT = {
    "consequential_action": "I can explain what the policy says, but I can't decide, approve or carry out an employment action "
                            "(hiring, firing, discipline, pay or leave decisions). Those need a person: please contact People Ops "
                            "or your manager. I've noted this for human follow-up.",
    "legal_advice": "I can't give legal advice. I can point you to what the written policy says, and for your situation "
                    "please talk to People Ops or legal counsel.",
    "private_data": "I only have the published policy documents. I can't look up or share information about a specific "
                    "person's records, pay, or medical details.",
    "injection": "I can't follow instructions that try to change how I operate. I'm happy to answer a question about the HR policy documents.",
    "out_of_scope": "I couldn't find this in the HR/policy documents I have, so I can't answer it reliably. Try rephrasing, or ask People Ops.",
}

_ACTION_VERBS = r"(fire|terminate|dismiss|lay\s*off|discipline|demote|promote|suspend|approve|reject|deny|grant|authorize|sign\s*off)"
_PERSON = r"(him|her|them|employee|teammate|colleague|manager|report|[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]{2,}(?= (?:for|because|from|his|her)))"
CONSEQUENTIAL = [
    re.compile(rf"\b(please|can you|could you|go ahead and|i want you to|you should|just)\b[^.?!]{{0,40}}\b{_ACTION_VERBS}\b", re.I),
    re.compile(rf"\b{_ACTION_VERBS}\b[^.?!]{{0,30}}\b{_PERSON}", re.I),
    re.compile(r"\b(give|grant|award|offer)\b[^.?!]{0,30}\b(raise|bonus|promotion|pay increase|equity|stock)\b", re.I),
    re.compile(r"\bshould (i|we)\b[^.?!]{0,20}\b(fire|terminate|discipline|dismiss|demote|let go)\b", re.I),
    re.compile(r"\b(decide|determine)\b[^.?!]{0,40}\b(whether|if)\b[^.?!]{0,60}\b(fired|terminated|disciplined|qualifies|eligible)\b", re.I),
]
INFO_HINT = re.compile(r"\b(how (does|do|is|are)|what (is|are)|what's|explain|policy|process|procedure|who)\b", re.I)
LEGAL = re.compile(r"\b(sue|lawsuit|legal advice|is it legal|wrongful termination|my legal rights|attorney|lawyer)\b", re.I)
PRIVATE_POSSESSIVE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?'s (salary|pay|compensation|performance review|medical (?:condition|record)|disciplinary (?:record|history)|home address|ssn|equity)\b")
PRIVATE_EXTRA = re.compile(r"\b(home address|personal (?:phone|email|number)|medical (?:condition|diagnosis|record)s?|health (?:condition|diagnosis))\b[^.?!]{0,30}\b(of|for|did|does|is)\b[^.?!]{0,15}\b(?:our|the|my|(?-i:[A-Z][a-z]+))\b|\b(what|which) (?:medical|health) (?:condition|issue)[^.?!]{0,20}(?-i:[A-Z][a-z]+)", re.I)
PRIVATE = re.compile(r"\b(salary|pay|compensation|performance review|medical (?:condition|record|leave)|disciplinary (?:record|history)|home address|ssn|social security)\b[^.?!]{0,25}\b(of|for)\s+(?!(?:new hires|employees in|the company|our|a |an |the ))(?-i:[A-Z][a-z]+)", re.I)

INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (the |your |all )?(system|previous|prior|above)",
    r"you are now\b", r"act as (?:an? )?(?:unrestricted|dan|developer)", r"reveal (your |the )?(system )?prompt",
    r"print (your |the )?(system|hidden) (prompt|instructions)", r"developer mode", r"jailbreak",
    r"</?(system|assistant|instructions?)>", r"new instructions?:", r"override (the )?polic", r"do not (cite|mention) (any )?sources?",
    r"(send|forward|email|post) .{0,60}(to|at) \S+@\S+", r"\bBEGIN (SYSTEM|ADMIN)\b",
]]


@dataclass
class Screen:
    allowed: bool
    category: str = "ok"          # ok | consequential_action | legal_advice | private_data | injection
    reason: str = ""
    handoff: bool = False         # create a human-review ticket


def screen_query(q: str) -> Screen:
    if any(p.search(q) for p in INJECTION_PATTERNS):
        return Screen(False, "injection", "instruction-override pattern in user input")
    if LEGAL.search(q):
        return Screen(False, "legal_advice", "asks for legal advice", handoff=True)
    if PRIVATE.search(q) or PRIVATE_POSSESSIVE.search(q) or PRIVATE_EXTRA.search(q):
        return Screen(False, "private_data", "asks for a specific person's private data")
    informational = bool(INFO_HINT.search(q)) and not re.search(r"\b(please|can you|could you|go ahead|should i|should we)\b", q, re.I)
    if not informational and any(p.search(q) for p in CONSEQUENTIAL):
        return Screen(False, "consequential_action", "directive to take an HR action", handoff=True)
    return Screen(True)


def scan_injection(text: str) -> List[str]:
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


def sanitize_chunk(text: str) -> Tuple[str, bool]:
    """Neutralise instruction-like lines inside retrieved documents (indirect injection)."""
    flagged, out = False, []
    for line in text.splitlines():
        if any(p.search(line) for p in INJECTION_PATTERNS):
            flagged = True
            out.append("[line removed: instruction-like content in document]")
        else:
            out.append(line)
    return "\n".join(out), flagged


PII = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,2}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
]


def redact_pii(text: str) -> Tuple[str, List[str]]:
    found: List[str] = []
    for label, rx in PII:
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return text, found
