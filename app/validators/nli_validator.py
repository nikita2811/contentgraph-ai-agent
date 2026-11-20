import logging
import os
import re
from functools import lru_cache
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Label = Literal["entailment", "neutral", "contradiction"]

MAX_PREMISE_CHARS = 12000
MAX_SENTENCES = 10


class SentenceNLI(BaseModel):
    index: int = Field(description="0-based index of the hypothesis")
    entailment_score: float = Field(ge=0, le=1)
    neutral_score: float = Field(ge=0, le=1)
    contradiction_score: float = Field(ge=0, le=1)


class BatchNLI(BaseModel):
    results: list[SentenceNLI]


_PROMPT = """You are a strict NLI (natural language inference) classifier.

PREMISE (research evidence):
{premise}

HYPOTHESES (numbered):
{hypotheses}

For EACH hypothesis, judge it against the premise only (ignore outside knowledge).
Give three scores between 0 and 1 that sum to about 1:
- entailment_score: premise clearly supports the hypothesis
- contradiction_score: premise clearly conflicts with the hypothesis
- neutral_score: premise neither supports nor conflicts (e.g. opinion, new info, transitions)

Return one result per hypothesis, using its index."""


@lru_cache(maxsize=1)
def _get_nli_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    return llm.with_structured_output(BatchNLI)


def _to_dict(item: SentenceNLI) -> dict:
    scores = {
        "entailment": item.entailment_score,
        "neutral": item.neutral_score,
        "contradiction": item.contradiction_score,
    }
    return {
        "label": max(scores, key=scores.get),
        "entailment_score": item.entailment_score,
        "neutral_score": item.neutral_score,
        "contradiction_score": item.contradiction_score,
    }


def classify_nli_batch(premise: str, hypotheses: list[str]) -> list[dict]:
    """Classify many hypotheses against one premise in a single LLM call."""
    if not hypotheses:
        return []

    numbered = "\n".join(f"[{i}] {h[:500]}" for i, h in enumerate(hypotheses))
    prompt = _PROMPT.format(premise=premise[:MAX_PREMISE_CHARS], hypotheses=numbered)

    neutral_default = {
        "label": "neutral",
        "entailment_score": 0.0,
        "neutral_score": 1.0,
        "contradiction_score": 0.0,
    }
    out = [dict(neutral_default) for _ in hypotheses]

    try:
        result = _get_nli_chain().invoke(prompt)
        for item in result.results:
            if 0 <= item.index < len(hypotheses):
                out[item.index] = _to_dict(item)
    except Exception:
        logger.exception("NLI check failed; defaulting to neutral")

    return out


def classify_nli(premise: str, hypothesis: str) -> dict:
    """Check if hypothesis follows from premise (single pair)."""
    return classify_nli_batch(premise, [hypothesis])[0]


def nli_label(premise: str, hypothesis: str) -> Label:
    return classify_nli(premise, hypothesis)["label"]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def validate_content_against_research(
    research: str,
    generated_content: str | list,
    contradiction_threshold: float = 0.6,
) -> dict:
    """
    Split generated content into sentences and check each against research.
    Returns overall verdict and per-sentence breakdown.
    """
    if isinstance(generated_content, str):
        sentences = _split_sentences(generated_content)
    elif isinstance(generated_content, list):
        sentences = [s.strip() for s in generated_content if len(s.strip()) > 20]
    else:
        raise TypeError(f"Expected str or list, got {type(generated_content)}")

    sentences = sentences[:MAX_SENTENCES]
    scored = classify_nli_batch(research, sentences)
    results = [{"sentence": s, **r} for s, r in zip(sentences, scored)]

    contradiction_count = sum(
        1 for r in results if r["contradiction_score"] > contradiction_threshold
    )
    passed = contradiction_count == 0

    return {
        "passed": passed,
        "contradiction_count": contradiction_count,
        "total_checked": len(results),
        "details": results,
        "verdict": "✅ Consistent" if passed else f"❌ {contradiction_count} contradictions found",
    }