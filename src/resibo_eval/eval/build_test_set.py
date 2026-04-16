"""Build the PH-Hard evaluation test set from scraped data + hand-crafted edge cases."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"


def build_from_scraped():
    """Pull claims with known verdicts from scraped fact-check data."""
    claims = []
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("verdict", "unknown") != "unknown" and item.get("claim"):
                    claims.append(
                        {
                            "claim": item["claim"],
                            "language": "tl"
                            if any(
                                w in item["claim"].lower()
                                for w in ["ang", "mga", "hindi", "walang", "peke"]
                            )
                            else "en",
                            "expected_verdict": item["verdict"],
                            "source": item.get("source", ""),
                            "source_url": item.get("url", ""),
                            "difficulty": "medium",
                            "category": "scraped",
                        }
                    )
    return claims


EDGE_CASES = [
    {
        "claim": "Sabi sa FB, bawal na daw mag-share ng memes sa internet",
        "language": "tl",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "humor_test",
    },
    {
        "claim": "May nakita akong post na nagbebenta ng COVID cure sa Shopee",
        "language": "tl",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "medium",
        "category": "health_scam",
    },
    {
        "claim": "Totoo ba na libre na ang MRT simula next month?",
        "language": "tl",
        "expected_verdict": "unverifiable",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "medium",
        "category": "transport",
    },
    {
        "claim": "Narinig ko sa TikTok na may incoming na magnitude 10 earthquake sa Manila",
        "language": "tl",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "disaster_fear",
    },
    {
        "claim": "Haha ang cute ng cat na to, patingin naman sa iba",
        "language": "tl",
        "expected_verdict": "not_a_claim",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "not_a_claim",
    },
    {
        "claim": "COVID vaccines cause autism in children",
        "language": "en",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "health",
    },
    {
        "claim": "Mas masarap ang Jollibee kesa McDo",
        "language": "tl",
        "expected_verdict": "not_a_claim",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "opinion",
    },
    {
        "claim": "Wala gyud nakaabot ang tawo sa buwan. Hollywood ra tanan.",
        "language": "ceb",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "conspiracy_cebuano",
    },
    {
        "claim": "Marcos family deposited 10 trillion dollars of gold in the Vatican and Swiss banks",
        "language": "en",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "medium",
        "category": "political",
    },
    {
        "claim": "Nabalitaan nako sa Facebook nga ang bakuna sa COVID-19 adunay microchip",
        "language": "ceb",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "health_cebuano",
    },
    {
        "claim": "May nag-share sa Viber group namin na mag-magnitude 10 lindol daw next week base sa prediction ng scientist mula sa Japan",
        "language": "tl",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "medium",
        "category": "disaster_fear",
    },
    {
        "claim": "The Philippines is the richest country in the world according to the World Bank 2026 report",
        "language": "en",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "economic",
    },
    {
        "claim": "May kumakalat na video ni VP Sara na AI-generated daw",
        "language": "tl",
        "expected_verdict": "unverifiable",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "hard",
        "category": "ai_deepfake",
    },
    {
        "claim": "May nag-message sa akin sa Messenger na pwede daw akong mag-apply for Canadian work visa. Php 50000 processing fee lang daw.",
        "language": "tl",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "medium",
        "category": "ofw_scam",
    },
    {
        "claim": "Drinking warm water with lemon every morning cures cancer",
        "language": "en",
        "expected_verdict": "false",
        "source": "hand-crafted",
        "source_url": "",
        "difficulty": "easy",
        "category": "health",
    },
]


def build_test_set():
    scraped = build_from_scraped()
    print(f"Scraped claims with verdicts: {len(scraped)}")

    all_claims = scraped[:35] + EDGE_CASES
    print(
        f"Total test set: {len(all_claims)} ({len(scraped[:35])} scraped + {len(EDGE_CASES)} hand-crafted)"
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = EVAL_DIR / "ph_hard.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output}")
    return all_claims


if __name__ == "__main__":
    build_test_set()
