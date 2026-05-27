"""
Regression guard for DeepEval metrics.

Persists a baseline of metric scores to scores_baseline.json and fails
the CI run if any metric drops by more than REGRESSION_TOLERANCE.

Usage
-----
    python evals/regression_guard.py --scores scores_latest.json
    python evals/regression_guard.py --scores scores_latest.json --update-baseline

Scores file format (produced by conftest.py after a pytest run)::

    {
      "faithfulness":      {"mean": 0.83, "min": 0.72, "pass_rate": 1.0},
      "answer_relevancy":  {"mean": 0.91, "min": 0.80, "pass_rate": 1.0},
      "keyword_coverage":  {"mean": 0.94, "min": 0.83, "pass_rate": 1.0}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).parent.parent / "scores_baseline.json"
REGRESSION_TOLERANCE = 0.05   # allow up to 5-point drop before failing


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def check_regression(latest: dict, baseline: dict) -> list[str]:
    failures = []
    for metric, stats in latest.items():
        if metric not in baseline:
            print(f"  [new]  {metric}: no baseline yet, will record.")
            continue
        for key in ("mean", "pass_rate"):
            new_val = stats.get(key)
            old_val = baseline[metric].get(key)
            if new_val is None or old_val is None:
                continue
            drop = old_val - new_val
            status = "✓" if drop <= REGRESSION_TOLERANCE else "✗"
            print(
                f"  [{status}] {metric}.{key}: "
                f"{old_val:.3f} → {new_val:.3f}  (Δ={-drop:+.3f})"
            )
            if drop > REGRESSION_TOLERANCE:
                failures.append(
                    f"{metric}.{key} dropped {drop:.3f} "
                    f"(baseline={old_val:.3f}, latest={new_val:.3f}, "
                    f"tolerance={REGRESSION_TOLERANCE})"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to latest scores JSON")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="If set, write latest scores as new baseline (use after a promoted run)",
    )
    args = parser.parse_args()

    latest = load_json(Path(args.scores))
    baseline = load_json(BASELINE_PATH)

    print("\n=== Regression check ===")
    if not baseline:
        print("  No baseline found — creating one from current run.")
        BASELINE_PATH.write_text(json.dumps(latest, indent=2))
        print(f"  Baseline saved to {BASELINE_PATH}")
        return

    failures = check_regression(latest, baseline)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(latest, indent=2))
        print(f"\n  Baseline updated: {BASELINE_PATH}")

    if failures:
        print("\n=== REGRESSION DETECTED ===")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\n  All metrics within tolerance. ✓")


if __name__ == "__main__":
    main()