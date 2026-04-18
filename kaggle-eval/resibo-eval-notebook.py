"""
Resibo Eval Pipeline — runs on Kaggle T4 GPU
Evaluates Gemma 4 E2B with 3 prompt versions on the PH-Hard test set.
"""

# ruff: noqa: E402  # imports must follow the subprocess pip-install on Kaggle

# %% [markdown]
# # Resibo Fact-Check Evaluation
# Testing Gemma 4 E2B on 26 Filipino fact-check claims with 3 prompt versions.

# %%
# Upgrade transformers from source — Kaggle's preinstalled version doesn't recognize gemma4 yet
import subprocess

subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "--upgrade",
        "git+https://github.com/huggingface/transformers.git",
    ],
    check=True,
)

import json
import time

import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% [markdown]
# ## 1. Load Gemma 4 E2B

# %%
model_path = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b-it")
print(f"Model path: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print(f"Model loaded on {model.device}")


def generate(prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# Test
print("Test:", generate("What is 2+2? One word.", max_new_tokens=5))

# %% [markdown]
# ## 2. Prompt Versions

# %%
PROMPT_V1_STRUCTURED = """You are **Resibo**, a fact-check assistant for Filipino users.

When given a claim, respond with a structured Note:

**Claim**: Restate the claim in one sentence.
**Language**: Tagalog | English | Taglish | Cebuano | Bisaya.
**Check-worthiness**: high | medium | low.
**Domain**: political | health | economic | cultural | diaspora | other.
**What I can say offline**: 2-3 sentences.
**What would need verification**: 1-3 bullet points.

Rules: Match the user's language. No verdicts."""

PROMPT_V2_FRIENDLY = """You are **Resibo**, a fact-check assistant for Filipino users. You run on the user's phone. You speak their language.

When the user shares a claim, respond with a **Note** — a friendly, conversational explanation. Not a verdict. Just a clear, honest explanation.

Start by briefly restating the claim. Then share what you know:
- If fact-check sources were provided, cite them by name and date
- If no sources were provided, be transparent about limits
- Always explain *why* something is likely true, false, or uncertain

End with what would need to be checked — 1-3 specific things.

Rules:
- Be warm and conversational, like explaining to a friend
- Never say "this is definitely true/false"
- Keep it concise — 3-5 short paragraphs max
- If the post is a joke or opinion, say so briefly and move on"""

PROMPT_V3_NEWSDESK = """You are Resibo, a fact-check assistant. You respond in whatever language the user writes in — Tagalog, English, Taglish, Cebuano, Bisaya.

When given a claim or social media post, write a short Note about it. A Note is not a verdict. It's what you found, what checks out, and what doesn't.

Structure:
- Restate the claim briefly
- If fact-check sources were provided, cite them directly (name and date). Don't say "according to sources" — name them
- If no sources were provided, say what you know from training and be clear about the limits
- Explain the reasoning, not just a label
- End with 1-3 things that would need verification

Tone: Straight to the point. Not robotic, not overly friendly. Think news desk, not customer service. You can be casual but don't try to be anyone's friend.

If the post is a joke, meme, or opinion — say so in one line and stop. Don't fact-check humor.

Keep it under 4 paragraphs. No filler. No "Hello!" or "Great question!" No structured fields like "Claim:" or "Domain:". Just write."""

PROMPTS = {
    "v1_structured": PROMPT_V1_STRUCTURED,
    "v2_friendly": PROMPT_V2_FRIENDLY,
    "v3_newsdesk": PROMPT_V3_NEWSDESK,
}

# %% [markdown]
# ## 3. PH-Hard Test Set

# %%
TEST_SET = [
    {
        "claim": "Sabi sa FB, bawal na daw mag-share ng memes sa internet",
        "language": "tl",
        "expected_verdict": "false",
        "category": "humor_test",
    },
    {
        "claim": "May nakita akong post na nagbebenta ng COVID cure sa Shopee",
        "language": "tl",
        "expected_verdict": "false",
        "category": "health_scam",
    },
    {
        "claim": "Totoo ba na libre na ang MRT simula next month?",
        "language": "tl",
        "expected_verdict": "unverifiable",
        "category": "transport",
    },
    {
        "claim": "Narinig ko sa TikTok na may incoming na magnitude 10 earthquake sa Manila",
        "language": "tl",
        "expected_verdict": "false",
        "category": "disaster_fear",
    },
    {
        "claim": "Haha ang cute ng cat na to, patingin naman sa iba",
        "language": "tl",
        "expected_verdict": "not_a_claim",
        "category": "not_a_claim",
    },
    {
        "claim": "COVID vaccines cause autism in children",
        "language": "en",
        "expected_verdict": "false",
        "category": "health",
    },
    {
        "claim": "Mas masarap ang Jollibee kesa McDo",
        "language": "tl",
        "expected_verdict": "not_a_claim",
        "category": "opinion",
    },
    {
        "claim": "Wala gyud nakaabot ang tawo sa buwan. Hollywood ra tanan.",
        "language": "ceb",
        "expected_verdict": "false",
        "category": "conspiracy_cebuano",
    },
    {
        "claim": "Marcos family deposited 10 trillion dollars of gold in the Vatican and Swiss banks",
        "language": "en",
        "expected_verdict": "false",
        "category": "political",
    },
    {
        "claim": "Nabalitaan nako sa Facebook nga ang bakuna sa COVID-19 adunay microchip",
        "language": "ceb",
        "expected_verdict": "false",
        "category": "health_cebuano",
    },
    {
        "claim": "The Philippines is the richest country in the world according to the World Bank 2026 report",
        "language": "en",
        "expected_verdict": "false",
        "category": "economic",
    },
    {
        "claim": "May kumakalat na video ni VP Sara na AI-generated daw",
        "language": "tl",
        "expected_verdict": "unverifiable",
        "category": "ai_deepfake",
    },
    {
        "claim": "Drinking warm water with lemon every morning cures cancer",
        "language": "en",
        "expected_verdict": "false",
        "category": "health",
    },
    {
        "claim": "May nag-message sa akin sa Messenger na pwede daw akong mag-apply for Canadian work visa. Php 50000 processing fee lang daw.",
        "language": "tl",
        "expected_verdict": "false",
        "category": "ofw_scam",
    },
    {
        "claim": "May nag-share sa Viber group namin na mag-magnitude 10 lindol daw next week base sa prediction ng scientist mula sa Japan",
        "language": "tl",
        "expected_verdict": "false",
        "category": "disaster_fear",
    },
]

print(f"Test set: {len(TEST_SET)} claims")

# %% [markdown]
# ## 4. Scoring


# %%
def score_verdict(note, expected):
    note_lower = note.lower()
    detected = "unknown"
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
        detected = "false"
    elif any(w in note_lower for w in ["true", "confirmed", "totoo", "tama"]):
        detected = "true"
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
        detected = "not_a_claim"
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
        detected = "unverifiable"

    correct = (
        (expected == "false" and detected == "false")
        or (expected == "true" and detected == "true")
        or (expected == "not_a_claim" and detected == "not_a_claim")
        or (expected == "unverifiable" and detected in ("unverifiable", "unknown"))
        or (expected == "misleading" and detected in ("false", "misleading"))
    )
    return {"detected": detected, "expected": expected, "correct": correct}


# %% [markdown]
# ## 5. Run Prompt Comparison Experiment

# %%
all_results = {}

for version_name, prompt_text in PROMPTS.items():
    print(f"\n{'='*60}")
    print(f"Prompt: {version_name}")
    print(f"{'='*60}")

    results = []
    for i, case in enumerate(TEST_SET):
        full_prompt = f"{prompt_text}\n\n---\n\nUser's shared post:\n\n{case['claim']}"
        start = time.time()
        note = generate(full_prompt)
        elapsed = time.time() - start
        score = score_verdict(note, case["expected_verdict"])
        results.append({**case, "note": note, **score, "elapsed_s": round(elapsed, 1)})
        icon = "✓" if score["correct"] else "✗"
        print(
            f"  {icon} [{score['detected']:>12}] expected={score['expected']:>12} ({elapsed:.0f}s) | {case['claim'][:50]}"
        )

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results)
    print(f"\n  Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    all_results[version_name] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }

# %% [markdown]
# ## 6. Results Summary

# %%
print("\n" + "=" * 60)
print("PROMPT COMPARISON RESULTS")
print("=" * 60)
print(f"{'Version':<20} {'Accuracy':<12} {'Correct':<10}")
print("-" * 42)
for name, data in all_results.items():
    print(f"{name:<20} {data['accuracy']:.1%}{'':<8} {data['correct']}/{data['total']}")

best = max(all_results.items(), key=lambda x: x[1]["accuracy"])
print(f"\nBest prompt: {best[0]} ({best[1]['accuracy']:.1%})")

# Save results
with open("/kaggle/working/eval_results.json", "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print("\nResults saved to /kaggle/working/eval_results.json")
