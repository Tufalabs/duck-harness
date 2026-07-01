# The Duck 🦆

This repository is a self-contained exposition of **The Duck**: a tool-using solver that plays [ARC-AGI-3](https://arcprize.org) games through TAAF (the Tufa ARC-AGI Framework). It bundles the harness source, one full benchmark run, and the interactive viewer for that run.

We discuss our solution in depth in our [technical write-up on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133), and compare it to other approaches in our [blog post](https://tufalabs.ai/research/duck-harness/). We showcase our solution in an episode of [Machine Learning Street Talk](https://x.com/MLStreetTalk/status/2072326433922297975?s=20), and discuss it in more detail [YouTube](https://www.youtube.com/watch?v=Vg6FBKTlfOw).


## Layout

| Path | What it is |
| --- | --- |
| `ARC3-Inference/` | The Duck itself — the inference harness. Tool-using solver, prompts, run artifacts, scoring, and the run **viewer**. |
| `tufa-arc-agi-framework/` | TAAF: the `Benchmark` / `GameAPI` execution framework the harness runs on. |
| `example-run/` | One complete benchmark run (25 official games × 20 passes) — what the viewer opens by default. |
| `taaf-duck-harness-kaggle-share.ipynb` | The Kaggle notebook that drives a run end-to-end (installs the runtime, loads the benchmark, plays the games). |

## View the example run

The viewer renders any run's per-game grids, actions, and model reasoning. It only needs the base Python deps — no GPU and no private packages:

```bash
cd ARC3-Inference
uv sync --locked   # Python 3.12 + uv; base deps only (~seconds)
make view          # opens the bundled ../example-run at http://127.0.0.1:8011
```

`make view` defaults to the `example-run/` shipped at the repo root. Point it at another run with `make view VIEW_RUN_DIR=/path/to/run`.

> `make install` does the same base sync **plus** the `server` extra (vLLM + Torch, multi-GB, GPU-only) — needed to actually run the harness, not to view.

## Run it yourself

The harness plays through a local vLLM server or OpenRouter — see [`ARC3-Inference/README.md`](ARC3-Inference/README.md). The Kaggle notebook at the repo root reproduces a full competition run on Kaggle hardware.
