"""Replicate the Resibo pipeline in Python for evaluation.

Uses the actual Gemma 4 model weights via HuggingFace transformers,
not a cloud API proxy. Same model that runs on the phone.
"""

import os
import time
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

_model = None
_tokenizer = None


def load_model(model_path: str = None):
    """Load Gemma 4 weights. Caches globally so it's loaded once."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if model_path is None:
        import kagglehub

        model_path = kagglehub.model_download(
            "google/gemma-4/transformers/gemma-4-e2b-it"
        )
        print(f"Model downloaded to: {model_path}")

    print(f"Loading model from {model_path}...")
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return _model, _tokenizer


def call_gemma(prompt: str, max_new_tokens: int = 512) -> str:
    """Run inference on local Gemma model."""
    import torch

    model, tokenizer = load_model()

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response_ids = outputs[0][input_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt not found: {path}")


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
    claim: str,
    prompt_version: str = "triage_system.md",
    use_evidence: bool = True,
) -> dict:
    """Run the full Resibo pipeline on a single claim."""
    system_prompt = load_prompt(prompt_version)

    search_query = ""
    evidence_text = ""
    citations = []

    if use_evidence:
        kw_prompt = load_prompt("keyword_extraction.md").replace("{INPUT}", claim[:500])
        search_query = (
            call_gemma(kw_prompt, max_new_tokens=50).strip().split("\n")[0].strip()
        )

        if search_query:
            pplx = call_perplexity(search_query)
            evidence_text = pplx["text"]
            citations = pplx["citations"]

    if evidence_text:
        full_prompt = f"{system_prompt}\n\n## Web research results\n\n{evidence_text}\n\n---\n\nUser's shared post:\n\n{claim}"
    else:
        full_prompt = f"{system_prompt}\n\n---\n\nUser's shared post:\n\n{claim}"

    note = call_gemma(full_prompt, max_new_tokens=512)

    return {
        "claim": claim,
        "search_query": search_query,
        "evidence_text": evidence_text,
        "citations": citations,
        "note": note,
        "prompt_version": prompt_version,
        "use_evidence": use_evidence,
    }
