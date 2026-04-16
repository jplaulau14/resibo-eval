"""
Run the Resibo evaluation pipeline with the actual Gemma 4 model.

Usage:
    export KAGGLE_API_TOKEN=your_kaggle_token
    export PERPLEXITY_API_KEY=your_perplexity_key

    # Quick eval (5 claims, current prompt)
    uv run python -m resibo_eval.eval.run_eval --experiment quick --max-claims 5

    # Prompt comparison (all 3 versions, no evidence, tests raw model quality)
    uv run python -m resibo_eval.eval.run_eval --experiment prompt_comparison --max-claims 10

    # Evidence ablation (with vs without Perplexity)
    uv run python -m resibo_eval.eval.run_eval --experiment evidence_ablation --max-claims 10

    # Full eval (all claims, current prompt, with evidence)
    uv run python -m resibo_eval.eval.run_eval --experiment full
"""

import argparse
import json
import time
from pathlib import Path

from .pipeline import call_gemma, load_model, call_perplexity, load_prompt
from .prompts import PROMPT_VERSIONS

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = Path(__file__).resolve().parents[4] / "experiments" / "results"


def load_test_set() -> list[dict]:
    path = EVAL_DIR / "ph_hard.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_verdict(note: str, expected: str) -> dict:
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
            "not true",
            "walang ebidensya",
        ]
    ):
        detected_verdict = "false"
    elif any(w in note_lower for w in ["true", "confirmed", "totoo", "tama"]):
        detected_verdict = "true"
    elif any(
        w in note_lower
        for w in [
            "not a claim",
            "joke",
            "meme",
            "opinion",
            "hindi ito factual",
            "biro",
            "not a factual",
            "lighthearted",
            "personal preference",
        ]
    ):
        detected_verdict = "not_a_claim"
    elif any(
        w in note_lower
        for w in [
            "unverifiable",
            "cannot verify",
            "hindi ma-verify",
            "walang sapat",
            "can't verify",
            "insufficient",
        ]
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

    return {
        "detected_verdict": detected_verdict,
        "expected_verdict": expected,
        "correct": correct,
        "has_sources": has_sources,
    }


def run_single_claim(
    claim_text: str, system_prompt: str, use_evidence: bool = True
) -> dict:
    """Run the pipeline on one claim with a given system prompt."""
    search_query = ""
    evidence_text = ""
    citations = []

    if use_evidence:
        kw_prompt = load_prompt("keyword_extraction.md").replace(
            "{INPUT}", claim_text[:500]
        )
        search_query = (
            call_gemma(kw_prompt, max_new_tokens=50).strip().split("\n")[0].strip()
        )

        if search_query:
            pplx = call_perplexity(search_query)
            evidence_text = pplx["text"]
            citations = pplx["citations"]

    if evidence_text:
        full_prompt = f"{system_prompt}\n\n## Web research results\n\n{evidence_text}\n\n---\n\nUser's shared post:\n\n{claim_text}"
    else:
        full_prompt = f"{system_prompt}\n\n---\n\nUser's shared post:\n\n{claim_text}"

    note = call_gemma(full_prompt, max_new_tokens=512)

    return {
        "search_query": search_query,
        "evidence_text": evidence_text[:500],
        "citations": citations,
        "note": note,
    }


def run_experiment(
    name: str,
    test_set: list[dict],
    system_prompt: str,
    use_evidence: bool = True,
    max_claims: int = 0,
) -> dict:
    claims = test_set[:max_claims] if max_claims > 0 else test_set
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Claims: {len(claims)}, Evidence: {use_evidence}")
    print(f"{'='*60}\n")

    results = []
    for i, case in enumerate(claims):
        claim_text = case["claim"]
        print(f"  [{i+1}/{len(claims)}] {claim_text[:60]}...")
        try:
            start = time.time()
            output = run_single_claim(claim_text, system_prompt, use_evidence)
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
            icon = "✓" if score["correct"] else "✗"
            print(
                f"    {icon} detected={score['detected_verdict']} expected={case.get('expected_verdict')} ({elapsed:.1f}s)"
            )
            print(f"    Note: {output['note'][:120]}...")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({**case, "error": str(e)})
        time.sleep(0.5)

    correct = sum(1 for r in results if r.get("correct"))
    total = len([r for r in results if "error" not in r])
    accuracy = correct / total if total > 0 else 0
    sourced = sum(1 for r in results if r.get("has_sources"))

    summary = {
        "experiment": name,
        "use_evidence": use_evidence,
        "total_claims": len(claims),
        "evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "sourced_notes": sourced,
        "sourced_rate": round(sourced / total, 3) if total > 0 else 0,
        "errors": len(claims) - total,
    }

    print(f"\n{'─'*40}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1%})")
    print(
        f"Source citation rate: {sourced}/{total} ({sourced/total:.1%})"
        if total
        else ""
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_file = RESULTS_DIR / f"{name}_{ts}.json"
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
        choices=["quick", "prompt_comparison", "evidence_ablation", "full"],
        default="quick",
    )
    parser.add_argument("--max-claims", type=int, default=10)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    print("Loading model...")
    load_model(args.model_path)

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test cases from PH-Hard")

    if args.experiment == "quick":
        system_prompt = load_prompt("triage_system.md")
        run_experiment(
            "quick_eval", test_set, system_prompt, max_claims=args.max_claims
        )

    elif args.experiment == "prompt_comparison":
        all_summaries = []
        for version_name, prompt_text in PROMPT_VERSIONS.items():
            summary = run_experiment(
                f"prompt_{version_name}",
                test_set,
                prompt_text,
                use_evidence=False,
                max_claims=args.max_claims,
            )
            all_summaries.append(summary)

        print(f"\n{'='*60}")
        print("PROMPT COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"{'Version':<20} {'Accuracy':<12} {'Sourced':<12}")
        print(f"{'─'*44}")
        for s in all_summaries:
            print(
                f"{s['experiment']:<20} {s['accuracy']:.1%}{'':<8} {s['sourced_rate']:.1%}"
            )
        best = max(all_summaries, key=lambda s: s["accuracy"])
        print(f"\nBest: {best['experiment']} ({best['accuracy']:.1%})")

    elif args.experiment == "evidence_ablation":
        system_prompt = load_prompt("triage_system.md")
        s1 = run_experiment(
            "with_evidence",
            test_set,
            system_prompt,
            use_evidence=True,
            max_claims=args.max_claims,
        )
        s2 = run_experiment(
            "without_evidence",
            test_set,
            system_prompt,
            use_evidence=False,
            max_claims=args.max_claims,
        )

        print(f"\n{'='*60}")
        print("EVIDENCE ABLATION RESULTS")
        print(f"{'='*60}")
        print(
            f"With evidence:    {s1['accuracy']:.1%} accuracy, {s1['sourced_rate']:.1%} sourced"
        )
        print(
            f"Without evidence: {s2['accuracy']:.1%} accuracy, {s2['sourced_rate']:.1%} sourced"
        )
        delta = s1["accuracy"] - s2["accuracy"]
        print(f"Evidence delta:   {delta:+.1%}")

    elif args.experiment == "full":
        system_prompt = load_prompt("triage_system.md")
        run_experiment("full_eval", test_set, system_prompt, max_claims=0)


if __name__ == "__main__":
    main()
