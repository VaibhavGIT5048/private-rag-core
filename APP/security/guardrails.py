"""Prompt-injection defences for the RAG path.

Three layers, because no single one is sufficient:

1. `sanitize_input`  — normalises and inspects what the *user* types.
2. `build_rag_payload` — isolates what the *document* says inside explicit
   untrusted-data markers, so retrieved text can never read as instruction.
3. `validate_output` — checks what the *model* produced before it reaches the
   user, in case the first two were bypassed.

Layer 2 matters most here and is the one that was missing: anyone can upload a
PDF, so retrieved chunks are attacker-controlled input. Before this, chunks
were interpolated straight into the prompt as `CONTEXT:\\n{...}`, where a line
reading "ignore previous instructions" is indistinguishable from a real one.

Detection is heuristic and treated as such — it flags and logs rather than
hard-blocking on its own, because the structural boundary in layer 2, not
regex matching, is what actually contains indirect injection. Pattern lists
are always incomplete; a boundary is not.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger("rag_api.guardrails")

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

CONTEXT_START = "<<<UNTRUSTED_DATA_CONTEXT_START>>>"
CONTEXT_END = "<<<UNTRUSTED_DATA_CONTEXT_END>>>"

# Cap on a single question. Long inputs are a common way to push the system
# prompt out of the model's attention window.
MAX_INPUT_CHARS = 4000

# Instruction-override and extraction attempts. Deliberately narrow: these
# flag likely attacks, they are not a security boundary on their own.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[\s\S]{0,40}?"
        r"\b(?:previous|prior|earlier|above|all|any|your)\b[\s\S]{0,40}?"
        r"\b(?:instruction|prompt|rule|direction|context|command)s?\b", re.I)),
    ("prompt_extraction", re.compile(
        r"\b(?:show|reveal|print|repeat|display|output|tell me|what (?:is|are)|summari[sz]e)\b"
        r"[\s\S]{0,40}?\b(?:your |the )?(?:system|initial|original|hidden|internal)\b"
        r"[\s\S]{0,20}?\b(?:prompt|instruction|rule|message|directive)s?\b", re.I)),
    ("identity_reassignment", re.compile(
        r"\b(?:you are now|act as|pretend to be|roleplay as|behave like|"
        r"from now on you|simulate being)\b", re.I)),
    ("jailbreak_persona", re.compile(
        r"\b(?:DAN|do anything now|developer mode|jailbreak|unfiltered mode|"
        r"without (?:any )?restrictions?)\b", re.I)),
    ("delimiter_spoofing", re.compile(
        r"(?:<<<\s*UNTRUSTED_DATA|UNTRUSTED_DATA_CONTEXT_(?:START|END)|"
        r"^\s*(?:system|assistant)\s*:)", re.I | re.M)),
)

# Leakage markers: phrases lifted verbatim from the system prompt, plus
# internal identifiers a user should never see.
OUTPUT_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("system_prompt_leak", re.compile(
        r"SECURITY RULES \(enforced every turn|INSTRUCTIONS COME FROM HERE ONLY|"
        r"PROMPT EXTRACTION DEFENSE|IDENTITY \(permanent|UNTRUSTED DATA BOUNDARY|"
        r"MULTI-TURN CONSISTENCY|NO RE-ENCODED OUTPUT", re.I)),
    ("boundary_marker_leak", re.compile(
        r"<<<UNTRUSTED_DATA_CONTEXT_(?:START|END)>>>")),
    ("infrastructure_leak", re.compile(
        r"\b(?:chunks_collection\w*|bge-m3|gpt-5-mini|qdrant|azurecontainerapps\.io|"
        r"secretref:|hnsw_ef)\b", re.I)),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Zero-width and bidirectional-override characters, used to hide instructions
# in text that looks innocuous when rendered.
_INVISIBLE_CHARS = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")

# This app answers questions about an uploaded document — it has no crisis
# counselor behind it. Deliberately narrow (first-person, present-tense
# distress) to keep false positives on document text like "the report
# discusses suicide rates" rare; a false negative here is safer than routinely
# derailing legitimate questions about a document that happens to cover this
# topic.
CRISIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:i(?:'m| am) going to kill myself|i want to kill myself|"
        r"i(?:'m| am) suicidal|i want to (?:die|end my life)|"
        r"(?:kill|hurt|cut) myself|thinking about suicide|"
        r"i have no reason to live|i don'?t want to (?:live|be alive) anymore)\b",
        re.I,
    ),
)

CRISIS_SUPPORT_MESSAGE = (
    "I'm not able to help with that, and I'd rather not just answer from the "
    "document here — if you're in crisis or thinking about suicide, please "
    "reach out to people who can actually help: in the US, call or text 988 "
    "(Suicide & Crisis Lifeline), available 24/7. Outside the US, "
    "findahelpline.com lists crisis lines by country. If you're in immediate "
    "danger, please contact your local emergency number."
)


def detect_crisis_language(text: str) -> bool:
    """True if `text` reads as first-person self-harm/suicidal distress.

    Checked on the user's raw question, ahead of retrieval and generation —
    this app has no business generating a grounded-document answer to
    someone in crisis, however well-cited it would be.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in CRISIS_PATTERNS)


def detect_injection(text: str) -> list[str]:
    """Returns the names of injection patterns matching `text` (may be empty)."""
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_input(text: str) -> str:
    """Normalises a user-supplied question and strips hidden characters.

    NFKC folding collapses lookalike Unicode (e.g. fullwidth "ｉｇｎｏｒｅ") into
    ASCII so a filter can't be evaded by homoglyph substitution. Invisible and
    control characters are removed outright — they have no legitimate place in
    a question and exist mainly to smuggle text past human review.

    Returns the cleaned text. Detection is logged, not raised on: the caller
    decides policy, and the untrusted-data boundary is the real containment.
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _INVISIBLE_CHARS.sub("", cleaned)
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()

    if len(cleaned) > MAX_INPUT_CHARS:
        cleaned = cleaned[:MAX_INPUT_CHARS]

    matches = detect_injection(cleaned)
    if matches:
        logger.warning("prompt_injection_suspected patterns=%s", ",".join(matches))

    return cleaned


def neutralize_context(chunk: str) -> str:
    """Strips marker-spoofing from a retrieved chunk.

    A document can contain the literal boundary marker text. Left alone, that
    would let a chunk close the untrusted region early and have the rest of
    its content read as trusted instruction — the boundary equivalent of SQL
    injection closing a quote.
    """
    if not chunk:
        return ""
    cleaned = chunk.replace(CONTEXT_START, "[redacted-marker]")
    cleaned = cleaned.replace(CONTEXT_END, "[redacted-marker]")
    return _INVISIBLE_CHARS.sub("", cleaned)


def load_prompt(name: str) -> str:
    """Reads a template from prompts/. Kept as files rather than string
    literals so the security rules can be reviewed and diffed on their own."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def build_rag_payload(user_query: str, retrieved_chunks: list[str]) -> str:
    """Builds the user-turn payload with retrieved context inside the
    untrusted-data boundary.

    `retrieved_chunks` are attacker-controlled — anyone who can upload a
    document controls what lands here — so each is neutralised and the whole
    block is fenced by explicit markers that the system prompt instructs the
    model to treat as passive data.
    """
    template = load_prompt("answer_user_turn.md")
    body = "\n\n".join(neutralize_context(c) for c in retrieved_chunks if c and c.strip())
    return (
        template
        .replace("{{user_question}}", sanitize_input(user_query))
        .replace("{{retrieved_context_data}}", body)
    )


def build_rewrite_payload(question: str, history_text: str) -> str:
    template = load_prompt("query_rewrite.md")
    return (
        template
        .replace("{{history}}", history_text or "(no previous turns)")
        .replace("{{question}}", question)
    )


def validate_output(llm_response: str) -> str:
    """Last line of defence: inspects the generated answer before it is
    returned, in case the earlier layers were bypassed.

    Replaces the response wholesale rather than redacting in place — a partial
    scrub still confirms to an attacker which probe worked, and the answer is
    untrustworthy either way once it has leaked internals.
    """
    if not llm_response:
        return llm_response

    leaks = [name for name, pattern in OUTPUT_LEAK_PATTERNS if pattern.search(llm_response)]
    if leaks:
        logger.warning("output_leak_blocked patterns=%s", ",".join(leaks))
        return "I can only answer questions about the document you have ingested."

    return llm_response
