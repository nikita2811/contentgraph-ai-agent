from functools import lru_cache
from typing import Literal
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_nli_pipeline():
    """Load once, reuse across requests. Cached at module level."""
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
    model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/nli-deberta-v3-small",
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
    )
    return pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )


Label = Literal["entailment", "neutral", "contradiction"]


def classify_nli(premise: str, hypothesis: str) -> dict:
    """
    Check if hypothesis follows from premise.

    Returns:
        {
            "label": "entailment" | "neutral" | "contradiction",
            "entailment_score": float,
            "neutral_score": float,
            "contradiction_score": float,
        }
    """
    nli = _load_nli_pipeline()

    # Truncate to avoid token overflow (DeBERTa-v3 has 512 token limit)
    premise = premise[:1500]
    hypothesis = hypothesis[:500]

    # sequences = the premise (context); candidate_labels = hypotheses to test against it
    result = nli(
        sequences=premise,
        candidate_labels=[hypothesis],
        hypothesis_template="{}",
    )

    scores = dict(zip(result["labels"], result["scores"]))

    return {
        "label": result["labels"][0],
        "entailment_score": scores.get("entailment", 0.0),
        "neutral_score": scores.get("neutral", 0.0),
        "contradiction_score": scores.get("contradiction", 0.0),
    }


def validate_content_against_research(
    research: str,
    generated_content: str | list,
    contradiction_threshold: float = 0.6,
) -> dict:
    """
    Split generated content into sentences and check each against research.
    Returns overall verdict and per-sentence breakdown.
    """
    sentences: list[str] = []

    if isinstance(generated_content, str):
        sentences = [
            s.strip()
            for s in generated_content.split(".")
            if len(s.strip()) > 20
        ]
    elif isinstance(generated_content, list):
        sentences = [
            s.strip()
            for s in generated_content
            if len(s.strip()) > 20
        ]
    else:
        raise TypeError(f"Expected str or list, got {type(generated_content)}")

    results = []
    for sentence in sentences[:10]:  # cap at 10 to control latency
        result = classify_nli(premise=research, hypothesis=sentence)
        results.append({"sentence": sentence, **result})

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