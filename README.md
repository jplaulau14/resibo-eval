# resibo-eval

> Evaluation harness and datasets for [Resibo](https://github.com/jplaulau14/resibo-android) — the Gemma 4, on-device, offline fact-check Note generator.

This repo holds the offline evaluation pipeline that validates every model and system change before it ships to the Android app.

## What's here

- **PH-Hard eval set** — curated Filipino claims across politics, health, economy, culture, and diaspora, with ground-truth labels, gold source citations, and adversarial hard-negatives
- **Eval harness** — runs the full agent loop (triage → retrieval → reasoning → Note) against a frozen checkpoint and scores:
  - Calibrated F1 on claim verdicts
  - Abstention precision (the Note's refusal to over-commit)
  - Source citation recall
  - Latency on Pixel 8 Pro p95
  - Multilingual consistency (Tagalog · English · Taglish · Cebuano · Bisaya)
- **Reports** — markdown + JSON outputs per run, checked in for historical diffs

## Target metrics

North-star: calibrated F1 ≥ 0.75 on PH-Hard, abstention precision ≥ 0.85, time-to-Note ≤ 8s p95.

## Related repos

- [`resibo-android`](https://github.com/jplaulau14/resibo-android) — the Android client
- [`resibo-train`](https://github.com/jplaulau14/resibo-train) — QLoRA fine-tuning pipeline

## Status

🚧 Early scaffold for the [Gemma 4 Good Hackathon](https://kaggle.com) — deadline **May 18, 2026**.
