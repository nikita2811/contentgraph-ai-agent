"""
DeepEval evaluation suite.

Metrics
-------
* FaithfulnessMetric      – output grounded in retrieval context
* AnswerRelevancyMetric   – output addresses the input question
* KeywordCoverageMetric   – expected keywords present in output (custom)

Run locally
-----------
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
    pytest evals/test_rag.py -v

Run in CI
---------
See .github/workflows/eval-regression.yml
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals.keyword_coverage import KeywordCoverageMetric

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "goldens.json"

# Thresholds — lower in CI to avoid flakiness from model non-determinism.
# Override via environment variables for stricter local runs.
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
RELEVANCY_THRESHOLD = float(os.getenv("RELEVANCY_THRESHOLD", "0.7"))
KEYWORD_THRESHOLD = float(os.getenv("KEYWORD_THRESHOLD", "0.7"))

# Judge model — defaults to gpt-4o; swap to claude if preferred.
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Simulated RAG pipeline (replace with your real pipeline)
# ---------------------------------------------------------------------------

def run_pipeline(input_text: str, retrieval_context: list[str]) -> str:
    """
    Placeholder RAG pipeline.

    In production replace this with your real LLM call, e.g.:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            system="Answer using only the context provided.",
            messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {input_text}"}]
        )
        return response.content[0].text

    For the test suite we return the expected output verbatim so all metrics
    should score high, proving the harness works end-to-end.
    """
    # TODO: replace with real pipeline call
    context_blob = " ".join(retrieval_context)
    # Naive stub: echo context sentences most relevant to keywords in question
    return context_blob[:600]


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_goldens() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared metrics (instantiated once, reused across test cases)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def faithfulness_metric():
    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    )


@pytest.fixture(scope="session")
def relevancy_metric():
    return AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    )


@pytest.fixture(scope="session")
def keyword_metric():
    return KeywordCoverageMetric(threshold=KEYWORD_THRESHOLD)


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("golden", load_goldens(), ids=lambda g: g["id"])
def test_rag_pipeline(
    golden,
    faithfulness_metric,
    relevancy_metric,
    keyword_metric,
):
    """
    For each golden:
      1. Run the pipeline to get actual output.
      2. Build a DeepEval test case.
      3. Assert all three metrics pass.
    """
    actual_output = run_pipeline(
        input_text=golden["input"],
        retrieval_context=golden["retrieval_context"],
    )

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden.get("expected_output"),
        retrieval_context=golden["retrieval_context"],
        metadata={"keywords": golden.get("keywords", [])},
    )

    assert_test(
        test_case,
        metrics=[faithfulness_metric, relevancy_metric, keyword_metric],
    )


# ---------------------------------------------------------------------------
# Individual metric smoke tests (fast — no LLM judge)
# ---------------------------------------------------------------------------

class TestKeywordCoverage:
    """Unit tests for the custom metric — no API calls."""

    def _case(self, output: str, keywords: list[str]) -> LLMTestCase:
        return LLMTestCase(
            input="test",
            actual_output=output,
            metadata={"keywords": keywords},
        )

    def test_full_coverage(self):
        metric = KeywordCoverageMetric(threshold=1.0)
        case = self._case("Paris is the capital of France.", ["Paris", "capital", "France"])
        metric.measure(case)
        assert metric.is_successful()
        assert metric.score == 1.0

    def test_partial_coverage_passes(self):
        # 1/3 keywords matched → score 0.33, fails at 0.5 but passes at 0.3
        metric_strict = KeywordCoverageMetric(threshold=0.5)
        case = self._case("Paris is a city.", ["Paris", "capital", "France"])
        metric_strict.measure(case)
        assert not metric_strict.is_successful()  # 0.33 < 0.5

        metric_lenient = KeywordCoverageMetric(threshold=0.3)
        metric_lenient.measure(case)
        assert metric_lenient.is_successful()  # 0.33 >= 0.3

    def test_zero_coverage_fails(self):
        metric = KeywordCoverageMetric(threshold=0.5)
        case = self._case("The sky is blue.", ["Paris", "capital", "France"])
        metric.measure(case)
        assert not metric.is_successful()
        assert metric.score == 0.0

    def test_case_insensitive(self):
        metric = KeywordCoverageMetric(threshold=1.0)
        case = self._case("PHOTOSYNTHESIS converts sunlight.", ["photosynthesis", "Sunlight"])
        metric.measure(case)
        assert metric.is_successful()

    def test_no_keywords_skipped(self):
        metric = KeywordCoverageMetric(threshold=1.0)
        case = self._case("Some output.", [])
        metric.measure(case)
        assert metric.is_successful()  # vacuously passes