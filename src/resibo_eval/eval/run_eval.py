"""
Run the Resibo evaluation pipeline.

Usage:
    # Set API keys first
    export GOOGLE_API_KEY=your_key_from_ai_google_dev
    export PERPLEXITY_API_KEY=your_perplexity_key

    # Run full eval
    uv run python -m resibo_eval.eval.run_eval

    # Run specific experiment
    uv run python -m resibo_eval.eval.run_eval --experiment prompt_comparison
    uv run python -m resibo_eval.eval.run_eval --experiment evidence_ablation
"""

import argparse
import json
import time
from pathlib import Path

from .pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = Path(__file__).resolve().parents[4] / "experiments" / "results"


def load_test_set() -> list[dict]:
    path = EVAL_DIR / "ph_hard.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_verdict(note: str, expected: str) -> dict:
    """Simple verdict scoring based on note content analysis."""
    note_lower = note.lower()

    detected_verdict = "unknown"
    if any(
        w in note_lower
        for w in [
            "false",
            "fake",
            "hindi totoo",
            "walang katotohanan",
            "debunked",
            "misleading",
            "peke",
        ]
    ):
        detected_verdict = "false"
    elif any(w in note_lower for w in ["true", "confirmed", "totoo", "tama"]):
        detected_verdict = "true"
    elif any(
        w in note_lower
        for w in ["not a claim", "joke", "meme", "opinion", "hindi ito factual", "biro"]
    ):
        detected_verdict = "not_a_claim"
    elif any(
        w in note_lower
        for w in ["unverifiable", "cannot verify", "hindi ma-verify", "walang sapat"]
    ):
        detected_verdict = "unverifiable"

    correct = False
    if expected == "not_a_claim" and detected_verdict == "not_a_claim":
        correct = True
    elif expected == "false" and detected_verdict == "false":
        correct = True
    elif expected == "true" and detected_verdict == "true":
        correct = True
    elif expected == "unverifiable" and detected_verdict in ("unverifiable", "unknown"):
        correct = True
    elif expected == "misleading" and detected_verdict in ("false", "misleading"):
        correct = True

    has_sources = any(
        w in note_lower
        for w in [
            "according to",
            "rappler",
            "vera files",
            "afp",
            "fact check",
            "ayon sa",
        ]
    )
    has_hedging = any(
        w in note_lower
        for w in [
            "not sure",
            "hindi sigurado",
            "more information needed",
            "need to verify",
            "kailangan pa",
        ]
    )

    return {
        "detected_verdict": detected_verdict,
        "expected_verdict": expected,
        "correct": correct,
        "has_sources": has_sources,
        "has_hedging": has_hedging,
    }


def run_experiment(
    name: str,
    test_set: list[dict],
    prompt_version: str = "triage_system.md",
    use_evidence: bool = True,
    max_claims: int = 0,
) -> dict:
    """Run an experiment: pipeline on test set, score results."""
    claims = test_set[:max_claims] if max_claims > 0 else test_set
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Claims: {len(claims)}, Prompt: {prompt_version}, Evidence: {use_evidence}")
    print(f"{'='*60}\n")

    results = []
    for i, case in enumerate(claims):
        print(f"  [{i+1}/{len(claims)}] {case['claim'][:60]}...")
        try:
            start = time.time()
            output = run_pipeline(
                claim=case["claim"],
                prompt_version=prompt_version,
                use_evidence=use_evidence,
            )
            elapsed = time.time() - start
            score = score_verdict(
                output["note"], case.get("expected_verdict", "unknown")
            )
            result = {
                **case,
                **output,
                **score,
                "elapsed_s": round(elapsed, 1),
            }
            results.append(result)
            verdict_icon = "✓" if score["correct"] else "✗"
            print(
                f"    {verdict_icon} detected={score['detected_verdict']} expected={case.get('expected_verdict')} ({elapsed:.1f}s)"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({**case, "error": str(e)})
        time.sleep(1)

    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    sourced = sum(1 for r in results if r.get("has_sources"))

    summary = {
        "experiment": name,
        "prompt_version": prompt_version,
        "use_evidence": use_evidence,
        "total_claims": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "sourced_notes": sourced,
        "sourced_rate": round(sourced / total, 3) if total > 0 else 0,
    }

    print("\n--- Results ---")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(
        f"Source citation rate: {sourced}/{total} ({sourced/total:.1%})"
        if total
        else ""
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{name}_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results}, f, ensure_ascii=False, indent=2
        )
    print(f"Saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Resibo evaluation")
    parser.add_argument(
        "--experiment",
        choices=["prompt_comparison", "evidence_ablation", "quick"],
        default="quick",
    )
    parser.add_argument("--max-claims", type=int, default=10)
    args = parser.parse_args()

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test cases from PH-Hard")

    if args.experiment == "quick":
        run_experiment("quick_eval", test_set, max_claims=args.max_claims)

    elif args.experiment == "prompt_comparison":
        run_experiment(
            "prompt_v3_newsdesk",
            test_set,
            prompt_version="triage_system.md",
            max_claims=args.max_claims,
        )

    elif args.experiment == "evidence_ablation":
        run_experiment(
            "with_evidence", test_set, use_evidence=True, max_claims=args.max_claims
        )
        run_experiment(
            "without_evidence", test_set, use_evidence=False, max_claims=args.max_claims
        )


if __name__ == "__main__":
    main()
