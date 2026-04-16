"""Replicate the Resibo pipeline in Python for evaluation.

Same flow as the Android app:
1. Gemma extracts search query
2. Perplexity searches
3. Gemma generates Note with evidence
"""

import os
import httpx
from pathlib import Path

PROMPTS_DIR = (
    Path(__file__).resolve().parents[4]
    / "resibo-android"
    / "app"
    / "src"
    / "main"
    / "assets"
    / "prompts"
)

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt not found: {path}")


def call_gemma(prompt: str, model: str = "gemma-4-e4b-it") -> str:
    """Call Gemma via Google's Gemini API (Gemma models are accessible there)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY env var (from ai.google.dev)")

    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return parts[0].get("text", "") if parts else ""


def call_perplexity(query: str) -> dict:
    """Call Perplexity API for fact-check evidence."""
    if not PERPLEXITY_API_KEY:
        return {"text": "", "citations": []}

    resp = httpx.post(
        PERPLEXITY_URL,
        headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"},
        json={
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fact-check research assistant. For the given claim, find relevant fact-checks and evidence. Be concise (3-4 sentences). Mention specific organizations and their ratings.",
                },
                {"role": "user", "content": query},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = data.get("citations", [])
    return {"text": text, "citations": citations}


def run_pipeline(
    claim: str, prompt_version: str = "triage_system.md", use_evidence: bool = True
) -> dict:
    """Run the full Resibo pipeline on a single claim."""
    system_prompt = load_prompt(prompt_version)

    search_query = ""
    evidence_text = ""
    citations = []

    if use_evidence:
        kw_prompt = load_prompt("keyword_extraction.md").replace("{INPUT}", claim[:500])
        search_query = call_gemma(kw_prompt).strip().split("\n")[0].strip()

        if search_query:
            pplx = call_perplexity(search_query)
            evidence_text = pplx["text"]
            citations = pplx["citations"]

    if evidence_text:
        full_prompt = f"{system_prompt}\n\n## Web research results\n\n{evidence_text}\n\n---\n\nUser's shared post:\n\n{claim}"
    else:
        full_prompt = f"{system_prompt}\n\n---\n\nUser's shared post:\n\n{claim}"

    note = call_gemma(full_prompt)

    return {
        "claim": claim,
        "search_query": search_query,
        "evidence_text": evidence_text,
        "citations": citations,
        "note": note,
        "prompt_version": prompt_version,
        "use_evidence": use_evidence,
    }
