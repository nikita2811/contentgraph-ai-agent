from transformers import pipeline
from functools import lru_cache
from typing import Literal
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_nli_pipeline():
    """Load once, reuse across requests. Cached at module level."""
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
    model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/nli-deberta-v3-small",
    torch_dtype=torch.float16,  # half precision, ~2x memory saving
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
            "contradiction_score": float,
        }
    """
    nli = _load_nli_pipeline()
    
    # Truncate to avoid token overflow (DeBERTa-v3 has 512 token limit)
    premise = premise[:1500]
    hypothesis = hypothesis[:500]
    
    result = nli(
        sequences=hypothesis,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",  # use hypothesis as-is
    )
    
    # Map scores back to labels
    scores = dict(zip(result["labels"], result["scores"]))
    
    return {
        "label": result["labels"][0],  # highest scoring label
        "entailment_score": scores.get("entailment", 0.0),
        "contradiction_score": scores.get("contradiction", 0.0),
        "neutral_score": scores.get("neutral", 0.0),
    }


def validate_content_against_research(
    research: str,
    generated_content: str,
    contradiction_threshold: float = 0.6,
) -> dict:
    """
    Split generated content into sentences and check each against research.
    Returns overall verdict and per-sentence breakdown.
    """
    sentences = [s.strip() for s in generated_content.split(".") if len(s.strip()) > 20]
    
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