# CodeAlign

Iterative post-training pipeline for a small coding LLM, using compiler/linter feedback as an automatic, execution-grounded reward signal for DPO.

## Overview

CodeAlign explores execution-grounded alignment for code generation. Instead of relying on static, human-labeled preference data, DPO preference pairs are generated automatically: candidate completions are run in a sandbox and scored on whether they compile and pass tests, with ties broken by static-analysis metrics (cyclomatic complexity, lint score).

The core question this project is testing: does combining execution feedback with static-analysis metrics avoid a known failure mode of execution-only rewards — models that learn to produce code which passes tests but is needlessly complex or unidiomatic ("spaghetti code")?

**Pipeline:** curated data (AST + lint validated) → SFT → DPO preference generation (execution + static analysis) → DPO training → evaluation (Base vs. SFT vs. DPO, including an ablation of execution-only vs. composite reward).

## Status

Early stage — Phase 0 (setup and scope) is complete. See the [Roadmap](#roadmap) below for progress.

## Scope (v1)

- **Language: Python only.** Restricting scope to a single language keeps sandboxed execution, AST validation, and evaluation tractable for a single-person project. This is a deliberate design choice made to ship a complete, well-evaluated pipeline rather than a partial multi-language one — not a shortcut. Extending to other languages (e.g. via `tree-sitter` for AST parsing and per-language sandboxes) is a natural extension, out of scope for v1.
- **Base model: [`Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).** Chosen because it is already code-pretrained, dense (single-GPU feasible), and well supported by the Hugging Face / TRL / PEFT stack. A general-purpose instruct model (e.g. Llama-3.1-8B-Instruct) may be added later purely as an ablation baseline, to measure how much starting from a code-specialized model actually matters.

## Compute budget

Documented up front so the project's resource footprint is explicit, not an afterthought.

- **Hardware target:** 1x GPU with 24 GB VRAM — RTX 3090 / 4090 locally, or L4 / RTX A5000 / RTX 3090 on a rental provider (RunPod Community Cloud pricing, ≈$0.30–$0.70/hr depending on GPU model, as of July 2026).
- **Training method:** QLoRA (4-bit) for both SFT and DPO, via TRL + PEFT + bitsandbytes. Full fine-tuning of a 7B model is out of scope for a single-GPU portfolio budget.
- **Estimated compute:** ~25–40 GPU-hours total, covering SFT, DPO preference-pair generation (sandboxed execution + scoring), DPO training, and evaluation of all three checkpoints. The range is wide on purpose — it accounts for debugging and re-runs, not just a single idealized pass.
- **Estimated cost:** ~$10–$30 total.
- These are planning estimates. They will be replaced with measured numbers once Phase 2 (SFT) produces real wall-clock data.

## Training data

- **License:** the base dataset must carry a permissive license (MIT or Apache-2.0) to avoid ambiguity around reuse of the resulting model and code. Candidates: [`HuggingFaceH4/CodeAlpaca_20k`](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) (MIT) or a filtered, permissively-licensed subset of [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) (Apache-2.0). Final choice to be confirmed and recorded here at the start of Phase 1.
- **Validation:** every sample is checked with `py_compile` (does it parse/compile) and `ruff check` (does it pass basic style/lint rules) before entering the SFT set. Samples that fail either check are discarded — see Phase 1.

## Roadmap

- [x] **Phase 0** — Setup & scope
- [ ] **Phase 1** — Data curation (AST validation + linting)
- [ ] **Phase 2** — Supervised fine-tuning (SFT)
- [ ] **Phase 3** — DPO preference-pair generation (execution feedback + static analysis)
- [ ] **Phase 4** — DPO training
- [ ] **Phase 5** — Evaluation (Base vs. SFT vs. DPO + reward-ablation)
- [ ] **Phase 6** — Interactive demo (side-by-side model comparison)
- [ ] **Phase 7** (stretch) — Rust execution daemon / lightweight RL (GRPO) comparison

## Project structure

```
codealign/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── main.py
├── .env.example
├── .gitignore
├── configs/            # SFT / DPO trainer configs (Phase 2 & 4)
├── data/
│   ├── raw/            # untouched source data
│   ├── curated/        # AST + lint validated SFT data (Phase 1)
│   └── preferences/    # chosen/rejected pairs for DPO (Phase 3)
├── src/
│   ├── data_curation/  # AST validation, linting (Phase 1)
│   ├── training/       # SFT + DPO trainers (Phase 2 & 4)
│   └── evaluation/     # benchmarks, ablation (Phase 5)
├── tests/
├── scripts/            # thin CLI wrappers per phase
└── docker/             # demo (Phase 6) / execution daemon (Phase 7)
```

## Setup

```bash
git clone <repo-url>
cd codealign
uv sync
cp .env.example .env   # fill in WANDB_API_KEY and HF_TOKEN
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). This is independent of the training-data license, which is documented separately above.
