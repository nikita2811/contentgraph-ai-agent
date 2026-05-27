"""
Custom keyword coverage metric for DeepEval.

Measures what fraction of expected keywords appear in the actual output.
Uses fuzzy matching (lowercased substring) so "photosynthesis" matches
"Photosynthesis", "photosynthesised", etc.
"""
from __future__ import annotations

from typing import Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class KeywordCoverageMetric(BaseMetric):
    """
    Measures the fraction of expected keywords present in the actual output.

    Parameters
    ----------
    threshold:
        Minimum coverage score [0, 1] required to pass.
    keywords:
        List of required keywords. If None the test case must carry
        `additional_metadata["keywords"]`.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        keywords: Optional[list[str]] = None,
    ) -> None:
        self.threshold = threshold
        self._static_keywords = keywords
        self.score = 0.0
        self.success = False

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------
    @property
    def __name__(self) -> str:  # noqa: D401
        return "Keyword Coverage"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def measure(self, test_case: LLMTestCase) -> float:
        keywords = self._resolve_keywords(test_case)

        if not keywords:
            self.score = 1.0
            self.success = True
            self.reason = "No keywords specified — skipping."
            return self.score

        output_lower = (test_case.actual_output or "").lower()
        hits = [kw for kw in keywords if kw.lower() in output_lower]
        misses = [kw for kw in keywords if kw.lower() not in output_lower]

        self.score = round(len(hits) / len(keywords), 4)
        self.success = self.score >= self.threshold
        self.reason = (
            f"{len(hits)}/{len(keywords)} keywords found "
            f"(score={self.score:.2f}, threshold={self.threshold}). "
            + (f"Missing: {misses}" if misses else "All keywords present.")
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_keywords(self, test_case: LLMTestCase) -> list[str]:
        if self._static_keywords:
            return self._static_keywords
        meta = getattr(test_case, "metadata", None) or getattr(test_case, "additional_metadata", None) or {}
        return meta.get("keywords", [])