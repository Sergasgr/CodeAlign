# CodeAlign

Iterative post-training pipeline for a small coding LLM, using compiler/linter feedback as an automatic, execution-grounded reward signal for DPO.

## Overview

CodeAlign explores execution-grounded alignment for code generation. Instead of relying on static, human-labeled preference data, DPO preference pairs are generated automatically: candidate completions are run in a sandbox and scored on whether they compile and pass tests, with ties broken by static-analysis metrics (cyclomatic complexity, lint score).

The core question this project is testing: does combining execution feedback with static-analysis metrics avoid a known failure mode of execution-only rewards — models that learn to produce code which passes tests but is needlessly complex or unidiomatic ("spaghetti code")?

**Pipeline:** curated data (syntax + complexity + lint validated) → SFT → DPO preference generation (execution + static analysis) → DPO training → evaluation (Base vs. SFT vs. DPO, including an ablation of execution-only vs. composite reward).

## Status

Phase 3 (preference generation) is the current focus. Phases 0–2 are complete; Phase 3 code is written and ready for execution. See the [Roadmap](#roadmap) below.

## Scope (v1)

- **Languages: Python, Java, C++, C#, JavaScript, TypeScript.** These map onto JetBrains' own primary IDE lineup (PyCharm, IntelliJ IDEA, CLion, Rider, WebStorm) — a deliberate choice, not an arbitrary list, and it spans genuinely different paradigms (dynamic scripting, statically-typed OOP, systems-level manual memory, gradually-typed web) rather than six near-identical languages. Earlier drafts of this project scoped v1 to Python only; that changed once the dataset (see below) turned out to be multi-language by construction and multi-language support became an explicit goal rather than an accident of the data.
- **Validation approach:** two signals apply uniformly across all six languages — syntax validity (via [`tree-sitter`](https://tree-sitter.github.io/tree-sitter/) grammars, which parse without needing a full compiler toolchain or resolvable imports — important since these are code snippets/diffs, not complete buildable projects; note this is a Concrete Syntax Tree, not an AST, so it's called syntax validation throughout rather than "AST validation") and cyclomatic complexity (via [`lizard`](https://github.com/terryyin/lizard), which already supports all six languages out of the box). Style/lint scoring is a best-effort third signal, applied only where a low-friction, pip-installable linter exists for v1: `ruff` for Python, `cpplint` for C++. The other four languages are gated on syntax + complexity alone for now — that's a real trade-off (documented here rather than silently applied), not an oversight; adding `eslint`/PMD-equivalent/`csharp` linters later means installing their native toolchains (Node, JDK, .NET SDK), which is a reasonable Phase 7-level addition (see Roadmap) rather than a Phase 1 blocker.
- **Duplication:** two distinct checks, not one. An internal-duplication / code-smell check flags copy-pasted blocks *within* a single sample (a DRY violation — is this one piece of code repeating itself). A separate MinHash/LSH near-duplicate index flags samples that are near-identical to *another sample already accepted* (dataset-level redundancy — protects against wasting training signal on repeated examples). They catch different problems; both are active.
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
- **Format — two prompt types, not one, both derived from CommitPackFT itself:** samples where `old_contents` is empty or trivial (≤3 meaningful lines — a new file) are framed as **write-from-spec**: instruction only, no code context, target is `new_contents`. Samples with substantial `old_contents` are framed as **in-context edit**: existing file + instruction, target is `new_contents`. The instruction itself prefers `message` (full commit body) over `subject` (short line) whenever it adds real detail.

  Why both, not just one: in-context editing is realistic IDE-assistant behavior ("fix/refactor this"), but Phase 5's evaluation benchmarks (HumanEval, MBPP) are entirely write-from-spec in shape — training exclusively on in-context edits would leave a real mismatch between the SFT/DPO training distribution and what the project's own evaluation measures. Splitting on `old_contents` size gets both formats from the one dataset already vetted for license and language quality, rather than reintroducing a second dataset (with Magicoder's OpenAI-terms and language-labeling problems) or hand-writing prompts. The split point (`NEW_FILE_LINE_THRESHOLD` in `config.py`) is a starting heuristic — Phase 1's report notebook tracks the resulting `edit`/`new_file` mix so it can be checked empirically rather than assumed.

  This replaces the markdown-fence extraction used for the previous dataset candidate — `new_contents` is already raw source, and each language's own config file (not a self-reported per-row label) is what determines its language, so it doesn't need the same defensive handling Magicoder's `lang` column did.
- **Validation:** every `new_contents` sample passes through the syntax + complexity (+ lint, where available) checks above before entering the SFT set. Samples that fail are discarded — see Phase 1.

## Roadmap

- [x] **Phase 0** — Setup & scope
- [x] **Phase 1** — Data curation (syntax validation + complexity + linting)
- [x] **Phase 2** — Supervised fine-tuning (SFT) — see [below](#phase-2--sft-qlora)
- [x] **Phase 3** — DPO preference-pair generation (execution feedback + static analysis) — see [below](#phase-3--generation-of-preferences-with-composite-reward-for-dpo)
- [ ] **Phase 4** — DPO training
- [ ] **Phase 5** — Evaluation (Base vs. SFT vs. DPO + reward-ablation)
- [ ] **Phase 6** — Interactive demo (side-by-side model comparison)
- [ ] **Phase 7** (stretch, not blocking) — see [below](#phase-7--stretch-extensions)

## Phase 2 — SFT (QLoRA)

Training a pre-trained base model on a curated set of (instruction, code) pairs to specialize it for coding — teaching output format and basic coding patterns, turning a general language model into a coding assistant. This is necessary, not optional: a pristine, well-validated dataset (Phase 1's syntax + complexity + lint gates) doesn't mean a base model will code well if simply prompted — instruction tuning is what actually adapts its behavior to the task.

**What SFT is *not* meant to do here:** it's not where this project's core research question gets answered. SFT establishes a reasonably competent starting point; DPO with the composite execution + static-analysis reward (Phase 3–4) is where the actual alignment experiment happens. Overinvesting in SFT hyperparameter search would spend the compute budget on the less interesting phase — see the golden-recipe decision below.

- **Trainer:** `trl.SFTTrainer` + QLoRA (4-bit NF4) on `Qwen2.5-Coder-7B-Instruct`, training only on the Phase 1 "pristine" subset.
- **Hyperparameters — a fixed "golden recipe", not a sweep.** Bayesian search (Optuna) was ruled out on purpose: a single 24GB GPU makes iterating multiple full SFT runs on a 7B model impractical, and that compute is better spent on DPO preference-pair generation, which is where the project's actual contribution lives. Standard-practice LoRA values are used instead — see `src/training/config.py` for the exact numbers (rank, alpha, dropout, learning rate, optimizer, epochs) and the reasoning behind each.
- **VRAM budget:** QLoRA's 4-bit base (~4.5GB vs. ~14GB in fp16) + gradient checkpointing + a `per_device_train_batch_size=2` / `gradient_accumulation_steps=8` split (effective batch size 16) keeps this inside 24GB — see `notebooks/02_sft_experiments_log.ipynb` for the measured VRAM footprint and W&B loss curves once a run completes.
- **Evaluation:** [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) — HumanEval (164 hand-written problems; `pass@1` is the fraction solved on the first attempt) — rather than a hand-rolled eval script. Deliberately run from the terminal against a saved checkpoint, not from inside `sft_trainer.py`: training and evaluation stay decoupled, and the same harness invocation gets reused for the DPO checkpoint in Phase 5 for a like-for-like comparison.
- **Checkpoint:** `checkpoints/sft/final_model` (adapter + tokenizer).

## Phase 3 — Generation of Preferences with Composite Reward for DPO

**Execution-Based Iterative DPO:** you don't need humans to label "Chosen" vs "Rejected." Using automated execution (sandbox) and statistical analysis (linters) to automatically generate preference pairs for DPO. Execution-based feedback provides a deterministic ground truth, eliminating the subjectivity and high cost of human annotation.

The pipeline loops over every prompt in the Phase 1 pristine dataset — generating two candidate completions from the SFT checkpoint (at `temperature=0.7` with `top_p=0.95`, via `num_return_sequences=2`), executing both inside a resource-limited Docker container (`network_disabled`, `mem_limit=128m`), and labelling the pair according to three cases:

1. **Model generates two variations of code for a problem**
2. **Run both in a sandbox**
3. The labelling decision:
   - **Case A — One passes, one fails.**
     Chosen: the passing one. Rejected: the failing one.
     DPO update: train the model to prefer the passing logic.
   - **Case B — Both fail:** discard the pair to prevent noisy gradients.
   - **Case C — Both pass.** Rank them by quality metrics: cyclomatic complexity and lint score (the composite reward signal). The superior code becomes chosen.

### Composite IDE Reward Signal

Moving beyond binary compilation checks to a multi-dimensional reward system. A binary reward (=1 if execution succeeds without runtime errors, 0 otherwise) encourages "spaghetti code" and doesn't guide the model toward good engineering practices. The solution is a **Composite Reward**:

$$R_{\text{total}} = w_1 R_{\text{exec}} + w_2 R_{\text{static}} + w_3 R_{\text{style}}$$

| Component | Signal | Weight | Purpose |
|-----------|--------|--------|---------|
| $R_{\text{exec}}$ | Binary (execution success) | `w₁ = 1.0` | Functional correctness |
| $R_{\text{static}}$ | Negative, proportional to cyclomatic complexity (continuous) | `w₂ = 0.1` | Code quality metrics — force simplicity and adherence to best practices |
| $R_{\text{style}}$ | Penalty for violating PEP-8 (ruff) / cpplint conventions | `w₃ = 0.2` | Style adherence |

It aligns the model's objective with the human objective: clean, maintainable, working code. Introducing a continuous penalty for high cyclomatic complexity prevents reward hacking and ensures the model produces maintainable, idiomatic code.

### Dual output (same pattern as Phase 1)

Phase 3 writes two files, mirroring Phase 1's `pristine_dataset.jsonl` / `report_dataset.jsonl` split:

- **`data/preferences/dpo_dataset.jsonl`** — clean pairs (`prompt` / `chosen` / `rejected` only), ready for `DPOTrainer` in Phase 4.
- **`data/preferences/dpo_report.jsonl`** — every prompt processed (including discarded pairs), with full metadata: case classification (A/B/C), composite scores, cyclomatic complexity, and lint errors for both chosen and rejected candidates, plus language. This is what `notebooks/03_preference_generation_report.ipynb` reads to produce the evidence graphs.

### Tracking

W&B logging follows the same `wandb.init` / `wandb.log` / `wandb.finish` pattern used in Phases 2 and 4. Logged metrics:
- Case distribution: `case_a`, `case_b`, `case_c`, `case_c_tied`
- Pair counts: `pairs_generated`, `pairs_discarded`
- Quality gap: `avg_chosen_score` vs. `avg_rejected_score`, `avg_chosen_cc` vs. `avg_rejected_cc`, `avg_chosen_lint_errors` vs. `avg_rejected_lint_errors`

The Case C percentage is the single most important metric of the project — if it's >15%, the composite reward is actively separating quality among working code, which is the core contribution beyond binary pass/fail.

### Implementation

- **Sandbox:** `DockerSandbox` — ephemeral containers with `network_disabled=True`, `mem_limit=128m`, `timeout` enforcement. Multi-language: Python, JavaScript, TypeScript, Java, C++, Go, Rust (matching Phase 1's language scope). Image: `docker/Dockerfile.executor` (Python 3.11-slim, non-root user, no pip — candidates that import third-party packages fail, which is intended).
- **Candidate generation:** `CandidateGenerator` — loads the SFT checkpoint (`checkpoints/sft/final_model`) with QLoRA (4-bit NF4), generates `N_CANDIDATES=2` completions per prompt. Code is extracted from markdown fences if present (`parse_code`).
- **Orchestrator:** `PreferenceOrchestrator` — wires generation → execution → scoring → labelling. Reuses Phase 1's `linter_check()` and `get_cyclomatic_complexity()` for the static-analysis components of the composite reward.
- **Report notebook:** `notebooks/03_preference_generation_report.ipynb` — case distribution (bar + pie), composite score histograms (chosen vs. rejected), cyclomatic complexity boxplots, lint error boxplots, per-language breakdown, qualitative samples.

## Phase 7 — stretch extensions

None of these block Phases 1–6. Listed here so the scope decisions behind them are explicit rather than assumed.

- **Rust execution daemon.** Formalize the sandboxed code-executor (used for DPO preference generation, Phase 3) as an async service in Rust (`tokio`, gRPC/REST API), running in an isolated container. Legitimate systems-engineering scope beyond Python — safe execution of untrusted code is a genuine problem, not busywork.
- **Lightweight RL comparison.** A `GRPOTrainer` run (TRL) using the same composite reward built for DPO, compared against the DPO result. Covers the "reinforcement learning" item from the original internship posting that DPO alone doesn't, at a fraction of the risk of the CUDA-kernel route considered earlier in this project's planning.
- **Hardware/inference-level demonstration.** If there's still an appetite to show low-level GPU understanding: export the DPO model to GGUF and benchmark latency/throughput across quantization levels with `llama.cpp`, or profile the training run itself with `torch.profiler`/Nsight to show where GPU time actually goes. Both demonstrate real hardware literacy without the risk of a hand-rolled kernel that's hard to defend line-by-line in an interview.
- **Docker image with native linter toolchains — and Go/Rust as full languages, not just tokenizer support.** `eslint` (Node), a PMD-equivalent (JDK), `clippy` (rustup), and `golangci-lint` (Go) aren't pip-installable — that's exactly why they were dropped from Phase 1's v1 scope rather than left half-working. A dedicated Docker image bundling those toolchains removes that constraint, and is the natural point to also bring Go and Rust in as fully-supported languages (tree-sitter grammars for both are already sketched, commented out, in `validators.py`) rather than partial citizens with parsing but no lint signal. Go and Rust each have their own JetBrains IDE (GoLand, RustRover) — extending language scope here keeps the "languages mirror JetBrains' IDE lineup" reasoning from Phase 0 consistent through to 8 languages instead of 6, if this phase is reached.

Not listed here because it's already addressed elsewhere: Unsloth (TRL's SFT/DPO speed-up) is an optional dependency group for Phases 2/4, not a Phase 7 item — see [Compute budget](#compute-budget) and `pyproject.toml`.

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
├── checkpoints/         # trained adapters (gitignored) — checkpoints/sft/, checkpoints/dpo/
├── data/
│   ├── raw/            # untouched source data
│   ├── curated/        # syntax + complexity + lint validated SFT data (Phase 1)
│   │   ├── pristine_dataset.jsonl  # accepted samples (SFT training data)
│   │   ├── rejected_dataset.jsonl  # filtered-out samples
│   │   └── report_dataset.jsonl    # all samples with full metadata (notebook)
│   └── preferences/    # chosen/rejected pairs for DPO (Phase 3)
│       ├── dpo_dataset.jsonl       # clean pairs for DPOTrainer
│       └── dpo_report.jsonl        # all results with full metadata (notebook)
├── notebooks/
│   ├── 01_curation_report.ipynb     # Phase 1 curation report (accepted/rejected, by language, by prompt type)
│   ├── 02_sft_experiments_log.ipynb # Phase 2 lab journal + evaluation harness runs
│   └── 03_preference_generation_report.ipynb # Phase 3 case distribution, quality metrics, samples
├── src/
│   ├── __init__.py
│   ├── data_curation/       # Phase 1
│   │   ├── config.py        # language/license whitelists, thresholds
│   │   ├── validators.py    # orchestrates the checks below -> check_code()
│   │   ├── linters.py       # ruff / cpplint / lizard (complexity)
│   │   ├── code_smells.py   # internal (within-sample) duplication
│   │   ├── minhash.py       # cross-sample near-duplicate index
│   │   └── main.py          # loads CommitPackFT, runs the pipeline
│   ├── preference_generation/ # Phase 3
│   │   ├── preference_generation_config.py  # model, sandbox, reward weights, W&B
│   │   ├── candidate_generator.py           # dual-temperature generation from SFT checkpoint
│   │   ├── docker_sandbox.py                # isolated execution (multi-language)
│   │   ├── preference_orchestrator.py       # A/B/C labelling + composite reward
│   │   └── main.py                          # pipeline entry point, W&B tracking
│   ├── training/             # Phase 2 & 4 — each phase owns its config.py,
│   │   ├── config.py         # same pattern as data_curation/config.py, not
│   │   ├── sft_trainer.py    # a shared configs/ folder (see Phase 2 notes)
│   │   └── dpo_trainer.py
│   └── evaluation/           # benchmarks, ablation (Phase 5)
├── tests/
├── scripts/             # thin CLI wrappers per phase, e.g. 01_curate_data.sh
└── docker/
    └── Dockerfile.executor  # Phase 3 sandbox image (python:3.11-slim, non-root, no pip)
```

## Setup

```bash
git clone <repo-url>
cd codealign
uv sync
cp .env.example .env   # fill in WANDB_API_KEY and HF_TOKEN

# Phase 1 — curate the dataset
scripts/01_curate_data.sh

# Phase 2 — SFT training
uv run python -m src.training.sft_trainer

# Phase 3 — generate DPO preference pairs
docker build -f docker/Dockerfile.executor -t codealign-executor:latest .
uv run python -m src.preference_generation.main

# Phase 4 — DPO training
uv run python -m src.training.dpo_trainer
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). This is independent of the training-data license, which is documented separately above.