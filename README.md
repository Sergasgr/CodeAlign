# CodeAlign

Iterative post-training pipeline for a small coding LLM, using compiler/linter feedback as an automatic, execution-grounded reward signal for DPO.

## Overview

CodeAlign explores execution-grounded alignment for code generation. Instead of relying on static, human-labeled preference data, DPO preference pairs are generated automatically: candidate completions are run in a sandbox and scored on whether they compile and pass tests, with ties broken by static-analysis metrics (cyclomatic complexity, lint score).

The core question this project is testing: does combining execution feedback with static-analysis metrics avoid a known failure mode of execution-only rewards — models that learn to produce code which passes tests but is needlessly complex or unidiomatic ("spaghetti code")?

**Pipeline:** curated data (syntax + complexity + lint validated) → SFT → DPO preference generation (execution + static analysis) → DPO training → evaluation (Base vs. SFT vs. DPO, including an ablation of execution-only vs. composite reward).

## Status

Early stage — Phase 0 (setup and scope) is complete, Phase 1 (data curation) is in progress. See the [Roadmap](#roadmap) below.

## Scope (v1)

- **Languages: Python, Java, C++, C#, JavaScript, TypeScript.** These map onto JetBrains' own primary IDE lineup (PyCharm, IntelliJ IDEA, CLion, Rider, WebStorm) — a deliberate choice, not an arbitrary list, and it spans genuinely different paradigms (dynamic scripting, statically-typed OOP, systems-level manual memory, gradually-typed web) rather than six near-identical languages. Earlier drafts of this project scoped v1 to Python only; that changed once the dataset (see below) turned out to be multi-language by construction and multi-language support became an explicit goal rather than an accident of the data.
- **Validation approach:** two signals apply uniformly across all six languages — syntax validity (via [`tree-sitter`](https://tree-sitter.github.io/tree-sitter/) grammars, which parse without needing a full compiler toolchain or resolvable imports — important since these are code snippets/diffs, not complete buildable projects) and cyclomatic complexity (via [`lizard`](https://github.com/terryyin/lizard), which already supports all six languages out of the box). Style/lint scoring is a best-effort third signal, applied only where a low-friction, pip-installable linter exists for v1: `ruff` for Python, `cpplint` for C++. The other four languages are gated on syntax + complexity alone for now — that's a real trade-off (documented here rather than silently applied), not an oversight; adding `eslint`/PMD-equivalent/`csharp` linters later means installing their native toolchains (Node, JDK, .NET SDK), which is a reasonable Phase 7-level addition (see Roadmap) rather than a Phase 1 blocker.
- **Base model: [`Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).** Chosen because it is already code-pretrained, dense (single-GPU feasible), and well supported by the Hugging Face / TRL / PEFT stack. A general-purpose instruct model (e.g. Llama-3.1-8B-Instruct) may be added later purely as an ablation baseline, to measure how much starting from a code-specialized model actually matters.

## Compute budget

Documented up front so the project's resource footprint is explicit, not an afterthought.

- **Hardware target:** 1x GPU with 24 GB VRAM — RTX 3090 / 4090 locally, or L4 / RTX A5000 / RTX 3090 on a rental provider (RunPod Community Cloud pricing, ≈$0.30–$0.70/hr depending on GPU model, as of July 2026).
- **Training method:** QLoRA (4-bit) for both SFT and DPO, via TRL + PEFT + bitsandbytes. Full fine-tuning of a 7B model is out of scope for a single-GPU portfolio budget.
- **Estimated compute:** ~25–40 GPU-hours total, covering SFT, DPO preference-pair generation (sandboxed execution + scoring), DPO training, and evaluation of all three checkpoints. Data curation (Phase 1) is CPU-bound and not part of this GPU-hour estimate. The range is wide on purpose — it accounts for debugging and re-runs, not just a single idealized pass.
- **Estimated cost:** ~$10–$30 total.
- These are planning estimates. They will be replaced with measured numbers once Phase 2 (SFT) produces real wall-clock data.

## Training data

- **Dataset: [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft).** Real commits from permissively-licensed GitHub repositories, filtered by the BigCode team to keep only commit messages that read as natural-language instructions — used to train OctoCoder in the [OctoPack paper](https://arxiv.org/abs/2308.07124). 702K samples across 277 languages, ~1.6 GB. Each sample carries an explicit per-sample `license` field, and content is real developer code (bug fixes, feature additions), not LLM-synthesized — this sidesteps the OpenAI-terms-of-use ambiguity that came up with the previously-considered `Magicoder-OSS-Instruct-75K` (GPT-3.5-generated) and is arguably a better fit for an IDE-assistant use case than isolated algorithmic puzzles.
- **License filter:** restricted to samples tagged `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`, `unlicense`, or `cc0-1.0`. The dataset's own license list also includes `agpl-3.0`, `lgpl-2.1`, `epl-1.0`, `mpl-2.0` and `unknown` — those are copyleft or ambiguous, not permissive in the strict sense, and are excluded even though the dataset card describes the whole set as "permissively licensed."
- **Language filter:** restricted to the six languages in scope (see above). Everything else (`yaml`, `json`, `markdown`, `html`, `css`, and 270+ others) is dropped — most of it isn't code in the sense this project cares about anyway.
- **Format:** each sample is a commit `subject`/`message` (used as the instruction) plus `old_contents`/`new_contents` (the file before/after). The training target is `new_contents`; `old_contents` may be included as context. This replaces the markdown-fence extraction approach used for the previous dataset candidate — `new_contents` is already raw source, and the `lang` field reflects the file's actual extension rather than a self-reported label, so it doesn't need the same defensive handling.
- **Validation:** every `new_contents` sample passes through the syntax + complexity (+ lint, where available) checks above before entering the SFT set. Samples that fail are discarded — see Phase 1.

## Roadmap

- [x] **Phase 0** — Setup & scope
- [ ] **Phase 1** — Data curation (syntax validation + complexity + linting)
- [ ] **Phase 2** — Supervised fine-tuning (SFT)
- [ ] **Phase 3** — DPO preference-pair generation (execution feedback + static analysis)
- [ ] **Phase 4** — DPO training
- [ ] **Phase 5** — Evaluation (Base vs. SFT vs. DPO + reward-ablation)
- [ ] **Phase 6** — Interactive demo (side-by-side model comparison)
- [ ] **Phase 7** (stretch) — Rust execution daemon / lightweight RL (GRPO) comparison / native linters for Java, C#, JS-TS (eslint, an MSBuild-based C# analyzer)

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
│   ├── curated/        # syntax + complexity + lint validated SFT data (Phase 1)
│   └── preferences/    # chosen/rejected pairs for DPO (Phase 3)
├── notebooks/
│   └── report.ipynb    # Phase 1 curation report (accepted/rejected breakdown)
├── src/
│   ├── __init__.py
│   ├── data_curation/  # syntax, complexity, linting (Phase 1)
│   ├── training/        # SFT + DPO trainers (Phase 2 & 4)
│   └── evaluation/      # benchmarks, ablation (Phase 5)
├── tests/
├── scripts/             # thin CLI wrappers per phase, e.g. 01_curate_data.sh
└── docker/              # demo (Phase 6) / execution daemon + multi-toolchain linting (Phase 7)
```

## Setup

```bash
git clone <repo-url>
cd codealign
uv sync
cp .env.example .env   # fill in WANDB_API_KEY and HF_TOKEN

# Phase 1 — curate the dataset
scripts/01_curate_data.sh
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). This is independent of the training-data license, which is documented separately above.