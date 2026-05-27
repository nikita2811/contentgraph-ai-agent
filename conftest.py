"""
conftest.py — shared pytest hooks for the eval suite.

After the test session finishes, writes a scores_latest.json summarising
each metric's mean score, min score, and pass rate across all test cases.
This file is consumed by regression_guard.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

SCORES_PATH = Path(__file__).parent / "scores_latest.json"

# Accumulate results keyed by metric name
_metric_scores: dict[str, list[float]] = defaultdict(list)
_metric_passed: dict[str, list[bool]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Hook: capture metric results after each test item
# ---------------------------------------------------------------------------

def pytest_runtest_logreport(report):
    """Extract DeepEval metric scores from the test item after it runs."""
    if report.when != "call":
        return

    # DeepEval attaches metric results to the test item via the plugin;
    # they appear in report.user_properties as ("deepeval_metric_<name>", {score, passed})
    for key, value in (report.user_properties or []):
        if not key.startswith("deepeval_metric_"):
            continue
        metric_name = key[len("deepeval_metric_"):]
        if isinstance(value, dict):
            score = value.get("score")
            passed = value.get("passed")
            if score is not None:
                _metric_scores[metric_name].append(float(score))
            if passed is not None:
                _metric_passed[metric_name].append(bool(passed))


# ---------------------------------------------------------------------------
# Hook: write summary after session ends
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    if not _metric_scores:
        return

    summary = {}
    for metric in _metric_scores:
        scores = _metric_scores[metric]
        passed = _metric_passed.get(metric, [])
        summary[metric] = {
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "pass_rate": round(sum(passed) / len(passed), 4) if passed else None,
            "n": len(scores),
        }

    SCORES_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[conftest] Scores written to {SCORES_PATH}")