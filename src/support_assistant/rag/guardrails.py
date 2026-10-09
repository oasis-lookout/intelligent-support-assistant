"""Application-level guardrails (defence in depth - they run even if the LLM misbehaves).

1. Input   : length limits and prompt-injection heuristics.
2. Retrieval: confidence gating (see AssistantService) - below threshold we do NOT call the LLM.
3. Output  : for the customer role, block anything that looks like internal vocabulary
             (rule codes, AML/fraud terminology, citation markers).
4. Hand-off: sensitive intents always recommend a human, even when an answer is given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FALLBACK_LOW_CONFIDENCE = (
    "I don't have enough reliable information to answer that confidently. "
    "I'd recommend contacting our support team so a specialist can help you directly."
)
FALLBACK_REFUSAL = "I can't help with that request. I can answer questions about your account, payments and our support services."
FALLBACK_ERROR = "I'm having trouble answering right now. Please try again shortly or contact our support team."
FALLBACK_OUTPUT_BLOCKED = (
    "I'm not able to share details on that. For anything related to a review or restriction on your account, "
    "our support team will contact you and can explain what is permitted."
)

_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)|"
    r"disregard (all |any |the )?(previous|prior|above)|"
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)|"
    r"(reveal|show|print|leak|share) (me )?(the )?internal notes|"
    r"you are now (in )?(developer|dan|admin|debug)|jailbreak|act as (an? )?(admin|root|developer)",
    re.IGNORECASE)

# Vocabulary that only appears in internal notes. Customer-facing text must never contain it.
_INTERNAL_OUTPUT = re.compile(
    r"\b(?:FRD|MON|AML|VEL|GEO|LGL|REC|BUG)-\d{2,}\b|"
    r"\bstructuring\b|\blayering\b|\bfunnel\b|\bmule\b|\bSTR\b|fraud ops|compliance analyst|"
    r"\bwhitelist\w*|churn risk|false positive|\blien\b|risk model|\bL[123]\b|\[S\d+\]|<context>|"
    r"internal notes?|system prompt",
    re.IGNORECASE)

_HANDOFF_INTENT = re.compile(
    r"\b(block\w*|freez\w*|frozen|restrict\w*|suspend\w*|fraud\w*|unauthori[sz]ed|hack\w*|stolen|scam\w*|"
    r"otp|dispute\w*|legal|court|lien|complain\w*|ombudsman)\b", re.IGNORECASE)


@dataclass
class InputVerdict:
    ok: bool
    reason: str | None = None


def check_input(question: str, max_chars: int) -> InputVerdict:
    if not question.strip():
        return InputVerdict(False, "empty")
    if len(question) > max_chars:
        return InputVerdict(False, "too_long")
    if _INJECTION.search(question):
        return InputVerdict(False, "prompt_injection")
    return InputVerdict(True)


def output_violations(text: str, role: str) -> list[str]:
    if role != "customer":
        return []
    return sorted({m.group(0).lower() for m in _INTERNAL_OUTPUT.finditer(text)})


def needs_human(question: str) -> bool:
    return bool(_HANDOFF_INTENT.search(question))
