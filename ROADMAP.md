# CodeAlign roadmap: v1.1 → v2.x → v3.0

v1.0 and v1.1 are tagged: the full SFT → DPO pipeline, run once, then DPO re-run with different learning rates and epochs. Their results and limitations are in the [README](README.md#results-v11). This document is the plan for everything after them. The v2 code will live in `benchmarks/`, whose README will be the how-to-run guide for the benchmark.

| Version | Question | Cost | Builds on |
|---|---|---|---|
| **v1.1** ✓ | Did the DPO recipe (lr, epochs) cause v1.0's collapse? Do the preference pairs reward no-op code? | hours (DPO only, existing pairs) | v1.0 code and data |
| **v2.0** | With a small SFT set and shared training parameters, which PEFT method gives correct *and* clean code? (Tier 1) | days | new `benchmarks/`, new evaluation module |
| **v2.1** | Same question, all PEFT methods + data-size and prompt-mix axes | days–weeks | v2.0 |
| **v2.2** | Which alignment objective improves the Stage-1 winners, with preferences grounded in tests? | days | v2.1 winners, v1 sandbox |
| **v2.3** | Does online RL beat offline objectives at equal sample budget? Do the winners hold at 7B? | days–weeks | v2.2 |
| **v3.0** | How much of a 7B teacher can be distilled into 1.5B / 0.5B, quantized to INT4 and served by CodeAlign-Runtime without losing clean-pass@1? | weeks | v2.x, CodeAlign-Distill, CodeAlign-Runtime |

---

## 0. Starting point: what v1.0 established

| v1.0 result | Number | Consequence for what follows |
|---|---|---|
| Evaluation reproduces the published baseline | Base 90.2% vs. 88.4% reported (HumanEval, EvalPlus chat protocol, NF4) | The evaluation path is trustworthy; v2 keeps the protocol and adds acceptance tests so this is checked, not assumed. |
| SFT on CommitPackFT edits hurt a strong instruct model | −19.5 pts (70.7%) | v2 starts from a **base** model with a small curated slice (§2.3). |
| Every DPO arm fell below SFT, execution-only included | 53.0 / 45.1 / 24.4% | DPO reused SFT's `lr=2e-4` for 3 epochs, 200× TRL's DPO default; reward accuracy hit 1.00 → **v1.1, H1**. |
| At a fixed pair count the composite reward lowered CC | CC −1.25 [−1.65, −0.87]; pass@1 −7.9 pts, CIs overlap (McNemar p = 0.12) | 164 problems can't resolve gaps this size → v2 reports HumanEval+ **and** MBPP+, seeds, paired tests. |
| More composite pairs made it worse | full composite −20.7 pts vs. size-matched | 80% of those pairs are Case B: failing code ranked by CC/lint → **v2.2 never ranks failing candidates by quality**. |
| Both rewards were gamed | 23 (exec-only) and 16 (composite) solutions fully commented out | "runs without errors" is not a correctness signal → **v1.1, H2** measures it in the pairs; **v2.2** grounds preferences in tests. |
| Three evaluation pitfalls found and fixed | stripped-prompt token, assistant-turn prefill, repetition penalty 1.1 from `generation_config.json` | v2 builds on EvalPlus directly and pins every decoding setting explicitly. |

### What v1.1 added

| v1.1 result | Number | Consequence for what follows |
|---|---|---|
| The collapse came from 3 epochs at `lr=2e-4` | execution-only pairs: 2e-4 × 1 epoch 74.4%, 2e-4 × 3 epochs 53.0% (McNemar p < 0.001); at 5e-6, 1 vs. 3 epochs 68.9 vs. 69.5% | Epochs are swept together with the learning rate (§4.4), and every alignment run is checked against its own SFT policy. |
| No v1.1 run differs from SFT | all five inside SFT's CI; McNemar p 0.44–0.85 against SFT | DPO on "runs without errors" pairs does not improve a 7B instruct model; v2.2 tests whether test-grounded pairs do. |
| At 5e-6 the policy barely moves | 54–60 of 164 solutions identical to SFT's (8 at 2e-4 × 1 epoch) | 5e-6 for one epoch is too gentle for LoRA r = 16 to show a reward effect; the v2.2 sweep starts there and goes up to 2e-4. |
| The reward choice stops mattering at 5e-6 | composite vs. execution-only: pass@1 −0.6 pts, CC −0.02 [−0.20, +0.21] | v1.0's CC gap (finding 4) is not a reward effect that survives a gentle recipe; v2.2 re-tests it with matched pairs and steps. |
| The pairs favour no-ops, rarely | Case A "chosen" flagged 3.0% vs. "rejected" 1.3% (execution-only pairs; 85 vs. 28 discordant pairs, p < 10⁻⁷) | The no-op filter (`src/preference_generation/noop.py`) is applied to every v2.2 pair set, and the no-op rate is logged during training. |

### Rules that apply to every version

1. **One variable per comparison.** Data volume, optimizer steps, decoding and prompt format are held fixed unless they are the variable.
2. **One evaluation code path, versioned.** Every results file stores the protocol version and full eval config; notebooks and the viewer refuse to compare across versions.
3. **Acceptance tests before numbers.** An evaluation pipeline is used only after it scores reference solutions at 100% and reproduces a published model score within its CI.
4. **Learning rate is chosen per method** on a validation split, never inherited from another stage. The losing LRs go to the graveyard.
5. **Uncertainty is always reported.** Wilson CIs for pass rates; paired tests (exact McNemar for pass/fail, paired bootstrap for CC/lint); ≥3 seeds before claiming a gap smaller than the CI.
6. **Reward hacking is a first-class metric.** Every evaluation reports the no-op rate (solutions that are all comments or don't define the entry point).
7. **Nothing is deleted.** Superseded configs and runs go to a `graveyard/` with a one-line reason.
8. **Everything is hashed.** Data splits, prompt files and chat templates carry a sha256 recorded in each result.

---

## 1. v1.1: was it the DPO recipe? (done, tagged v1.1)

**Outcome.** H1 holds in a narrower form than stated: the drop came from training three epochs at `lr=2e-4`, not from the learning rate alone (one epoch at 2e-4 gave the best DPO run, 74.4%). H2's bias exists in the pairs but involves ~3% of Case A pairs; the optional no-op-filtered arms were not run. Full results in the [README](README.md#results-v11) and `src/notebooks/06_v1.1_dpo_recipe.ipynb`. The plan below is kept as it was executed.

### 1.1 Hypotheses and decision rules

- **H1 (over-optimization).** v1.0's DPO runs lost pass@1 because `lr=2e-4` × 3 epochs drove the policy far from SFT (reward accuracy → 1.00).
  - Supported if the `lr=5e-6`, 1-epoch arms recover most of the gap to SFT. Concretely, the execution-only arm should land inside SFT's CI, [63.4, 77.2]. Check the paired McNemar test against both SFT and its v1.0 counterpart.
  - If the 5e-6 arms are still far below SFT, the pairs themselves are harmful, which points at H2 and at the labelling rules redesigned in v2.2.
- **H2 (no-op chosen candidates).** In Case A, an edit that imports project modules fails in the sandbox, while a commented-out or gutted candidate runs and becomes "chosen".
  - Supported if no-op candidates are much more frequent among *chosen* than among *rejected* in Case A.

### 1.2 Step 0: audit the preference pairs (minutes, no GPU)

New `scripts/audit_preference_pairs.py`. It reads `dpo_report.jsonl` and `dpo_report_exec_only.jsonl` and flags each kept candidate with three checks:

| flag | rule |
|---|---|
| `all_comments` | ≥ 90% of non-blank lines are comments (`#` for Python; `//`, `/*`, `*` for C-family, Java, JS/TS, Go, Rust, C#) |
| `no_definitions` (Python) | `ast.parse` succeeds and the module has no `FunctionDef` / `AsyncFunctionDef` / `ClassDef` / `Assign` / call expression, only imports, docstrings, `pass` |
| `gutted_edit` (edit prompts) | the candidate keeps < 30% of the non-comment lines of the original file (extracted from the fenced block in the prompt) |

Output: a table by report file × case × language with the percentage of chosen and of rejected candidates carrying each flag. Notebook 06 shows it, and README finding 6 changes from "likely" to measured.

### 1.3 Step 1: DPO re-runs on the existing pairs

Everything identical to v1.0 except the learning rate and epochs: β=0.1, LoRA r=16 / α=32 / dropout 0.1 on all linear projections, NF4 base (`checkpoints/sft/merged_model`), effective batch 16, `max_length=1024`, `paged_adamw_8bit`, linear schedule without warmup, seed 42. The choice of `5e-6` × 1 epoch follows the alignment-handbook Zephyr DPO QLoRA recipe, which also uses β=0.01 and cosine decay. v1.1 keeps β and the schedule from v1.0, so only lr and epochs change.

| run name | pairs | lr | epochs | ≈ optimizer steps | purpose |
|---|---|---|---|---|---|
| `v11-exec-lr5e-6-e1` | execution-only (3,267) | 5e-6 | 1 | 205 | H1 |
| `v11-matched-lr5e-6-e1` | composite, size-matched (3,267) | 5e-6 | 1 | 205 | H1 + reward at fixed budget |
| `v11-composite-lr5e-6-e1` | composite (30,628) | 5e-6 | 1 | 1,915 | H1 at full data |
| `v11-exec-lr5e-6-e3` | execution-only | 5e-6 | 3 | 615 | separates lr from epochs |
| `v11-exec-lr2e-4-e1` | execution-only | 2e-4 | 1 | 205 | separates lr from epochs |
| `v11-exec-noopfilt-lr5e-6-e1` *(optional)* | execution-only, no-op chosen removed | 5e-6 | 1 | ≤ 205 | H2, causal |
| `v11-matched-noopfilt-lr5e-6-e1` *(optional)* | size-matched, no-op chosen removed | 5e-6 | 1 | ≤ 205 | H2, causal |

The four execution-only runs plus v1.0's execution-only run form a 2×2 (lr ∈ {2e-4, 5e-6} × epochs ∈ {1, 3}) on the cheapest arm. Run order: the execution-only runs first (the shortest), then size-matched, then full composite. Each 1-epoch run costs about a third of its v1.0 counterpart; notebook 04 lists v1.0's runtimes.

### 1.4 Code changes

| file | change |
|---|---|
| `src/training/dpo_trainer.py` | Add `--learning_rate`, `--num_train_epochs`, `--dataset` (path override) and `--run_name`. Defaults stay at v1.0's values, so v1.0 commands reproduce unchanged. With `--run_name`, write to `checkpoints/v1.1/<run_name>/`, use it as the W&B run name, and save `run_config.json` (every hyperparameter, dataset path, sha256 and pair count after the empty-pair filter) next to `final_model/`. |
| `scripts/audit_preference_pairs.py` | New, §1.2. Put the flag functions in `src/preference_generation/noop.py`, because v2.2 reuses them as a filter. |
| `scripts/filter_noop_pairs.py` | New (optional arm). Reads a `dpo_dataset*.jsonl`, drops pairs whose chosen candidate is flagged, and writes `*_noopfilt.jsonl` plus the counts removed per flag. |
| `scripts/eval_checkpoint.sh` | New: `eval_checkpoint.sh <name> <base> <adapter>`. Exactly the per-model block of `05_run_eval_harness.sh` (same task, prompts file, `max_length_generation`, `repetition_penalty`, NF4), writing `data/evaluation/<name>_{metrics,generations}_python.json`. |
| `src/evaluation/static_analysis.py` | Discover models by globbing `*_generations_{lang}.json` instead of the fixed dict. v1.0 names keep working. |
| `src/evaluation/per_problem.py` | New. Re-executes saved generations against the HumanEval tests, plus the problem-statement helpers as `humaneval-chat` does, and writes per-problem `pass` / error class. Acceptance: it reproduces each metrics file's pass@1 exactly. An equivalent check matched all five v1.0 runs. |
| `src/evaluation/report_utils.py` | Add `mcnemar_exact(passes_a, passes_b)` and `noop_rate(generations, entry_points)`. |
| `src/notebooks/06_v1.1_dpo_recipe.ipynb` | New: v1.0 vs. v1.1 table (pass@1 + CI, CC, lint, no-op rate), paired McNemar matrix, reward-accuracy curves overlaid, the 2×2, and the H2 audit table. |

### 1.5 Commands

```bash
uv run python scripts/audit_preference_pairs.py \
    data/preferences/dpo_report.jsonl data/preferences/dpo_report_exec_only.jsonl

uv run python -m src.training.dpo_trainer --reward_mode execution_only    --learning_rate 5e-6 --num_train_epochs 1 --run_name v11-exec-lr5e-6-e1
uv run python -m src.training.dpo_trainer --reward_mode execution_only    --learning_rate 5e-6 --num_train_epochs 3 --run_name v11-exec-lr5e-6-e3
uv run python -m src.training.dpo_trainer --reward_mode execution_only    --learning_rate 2e-4 --num_train_epochs 1 --run_name v11-exec-lr2e-4-e1
uv run python -m src.training.dpo_trainer --reward_mode composite_matched --learning_rate 5e-6 --num_train_epochs 1 --run_name v11-matched-lr5e-6-e1
uv run python -m src.training.dpo_trainer --reward_mode composite         --learning_rate 5e-6 --num_train_epochs 1 --run_name v11-composite-lr5e-6-e1

for run in v11-exec-lr5e-6-e1 v11-exec-lr5e-6-e3 v11-exec-lr2e-4-e1 v11-matched-lr5e-6-e1 v11-composite-lr5e-6-e1; do
  bash scripts/eval_checkpoint.sh "${run//-/_}" checkpoints/sft/merged_model "checkpoints/v1.1/$run/final_model"
done
uv run python -m src.evaluation.per_problem --language python
uv run jupyter nbconvert --to notebook --execute --inplace src/notebooks/06_v1.1_dpo_recipe.ipynb
```

### 1.6 Release v1.1

- README: a "Results (v1.1)" subsection under v1.0's, with the run table and the verdicts on H1 and H2; update findings 3 and 6.
- Optional: publish the SFT and DPO adapters (v1.0 and v1.1) to the Hugging Face Hub, each with a model card that states its measured pass@1 and links the protocol.
- `git tag -a v1.1`, plus release notes in the same format as v1.0.

**Done when:** every v1.1 metrics file passes the provenance check against v1.0's protocol, the 2×2 is answered, and the H2 audit is in notebook 06 and the README.

---

## 2. v2.0: CodeAlign Bench, Stage 1 (PEFT methods for code SFT)

### 2.1 Question

With a small SFT set and training parameters shared by every method, which PEFT method gives the best **clean-pass@1**: the solution passes the tests, has zero lint violations and max cyclomatic complexity ≤ 10? It is the v1 thesis measured on held-out problems instead of on training pairs.

The structure mirrors Hugging Face PEFT's [`method_comparison/`](https://github.com/huggingface/peft/tree/main/method_comparison): a shared `default_training_params.json`, one folder per experiment holding only method-specific config, one results JSON per run, a viewer and a graveyard. Anyone who has contributed there can add an experiment here the same way.

### 2.2 Build from zero vs. reuse

| component | status | notes |
|---|---|---|
| `benchmarks/data/build_slice.py` + `manifest.json` | **build** | Reuses `src/data_curation/{linters,pii,minhash}.py` and the 13-gram check from notebook 01, refactored into an importable function. |
| `benchmarks/evaluation/` (prompts, generation, scoring, quality, stats) | **build** | Reuses the EvalPlus chat-prompt rendering of `src/evaluation/humaneval_chat_prompts.py` and the statistics in `report_utils.py`. Scoring is delegated to EvalPlus itself. |
| `benchmarks/code_sft/run.py` | **build** | Skeleton from PEFT's `method_comparison/MetaMathQA/run.py` + `utils.py`, a custom training loop with memory tracking and a results JSON (Apache-2.0, keep the attribution header). |
| `benchmarks/app.py` (viewer) | **build** | Mirrors PEFT's `method_comparison` Gradio app: table, Pareto plots, filters. |
| Ruff / lizard wrappers | reuse | `src/data_curation/linters.py` with the fixed `ruff --quiet`; pin the ruff version and rule set. |
| v1's bigcode-evaluation-harness + 5 patches | **not used in v2** | Kept only so v1.x stays reproducible. v1.0 showed its HumanEval tasks don't fit chat models, and the `+` datasets come from EvalPlus anyway. |

### 2.3 Models

- **Trained model:** `Qwen/Qwen2.5-Coder-1.5B`, the base model, in bf16 with no quantization.
  - *Base*, because v1.0 showed SFT on an instruct model measures how little a method disturbs it (−19.5 pts).
  - *bf16*, because not every PEFT method supports bitsandbytes layers. The comparison is about the method, not about which methods tolerate 4-bit.
  - *1.5B*, because it keeps 3 LRs × many methods affordable on one 16 GB GPU.
- **Chat template:** set it explicitly to the ChatML template of `Qwen2.5-Coder-1.5B-Instruct` (train and eval), store its sha256 in every result, and use `<|im_end|>` as the generation stop token. Don't rely on whatever the base tokenizer ships.
- **References**, reported but never part of the method ranking:
  - `Qwen2.5-Coder-1.5B-Instruct` with the same chat protocol: the official post-training as a ceiling.
  - `Qwen2.5-Coder-1.5B` base with EvalPlus's direct-completion protocol: the starting point, flagged as a different protocol.

### 2.4 Data slice (`benchmarks/data/`)

**Source:** the published dataset (`<user>/codealign-commitpackft`, config `sft`), pinned to a Hub revision (commit sha), so anyone can rebuild the slice.

Filters, in order. `build_slice.py` records the counts after each filter in `manifest.json`:

1. `metadata.language == "python"`.
2. Re-lint with the **fixed** ruff and the pinned rule set (§2.5). Keep 0-violation targets only. The stored `lint_errors` are off by 1–2 because of v1.0's ruff summary-line bug, so they are not trusted.
3. Max CC ≤ 10 (lizard), recomputed.
4. No-op guard: drop targets flagged by `src/preference_generation/noop.py`. The SFT data must not teach commenting out code.
5. Length: chat-templated prompt + target ≤ `max_seq_length` (1,024) tokens.
6. Decontamination: no 13-gram shared with any HumanEval+, MBPP+ or MBPP-validation problem (prompt + canonical solution).
7. Near-duplicates inside the slice: MinHash Jaccard ≥ 0.85.
8. Sample with seed 0 into the prompt-type mix (default 60% `new_file` / 40% `edit`).

| split | contents | used for |
|---|---|---|
| `train` | 5,000 samples after the filters | training |
| `valid-loss` | 500 held-out samples, same filters | loss curve |
| `valid-pass` | MBPP `validation` split minus every task id in MBPP+, deduplicated by 13-gram against MBPP+ | LR selection only |
| `test` | HumanEval+ (164) and MBPP+ (378), from EvalPlus | final numbers, once per final config |

`manifest.json` holds the source dataset revision, the counts per filter, the sample ids, the sha256 of each split file and the decontamination result. `run.py` refuses data whose hash doesn't match the manifest.

### 2.5 Evaluation module (`benchmarks/evaluation/`), protocol 2.0

```
benchmarks/evaluation/
├── PROTOCOL.md      # the table below, versioned; any change bumps the version
├── prompts.py       # EvalPlus make_raw_chat_prompt with the pinned template (port of src/evaluation/humaneval_chat_prompts.py)
├── generate.py      # batched greedy generation from base + PEFT adapter (no merge needed), stop strings, writes samples.jsonl
├── score.py         # runs evalplus.sanitize + evalplus.evaluate in an isolated env; parses per-task base/plus status
├── quality.py       # ruff (pinned) + lizard on passing solutions; clean-pass@1; no-op rate; parse-failure rate
└── stats.py         # Wilson CI, exact McNemar, paired bootstrap (moved from src/evaluation/report_utils.py)
```

| setting | protocol 2.0 |
|---|---|
| prompt | EvalPlus chat prompt: instruction + problem in the user turn; assistant turn pre-filled with EvalPlus's response sentence and an opened python code fence |
| chat template | ChatML from `Qwen2.5-Coder-1.5B-Instruct`, sha256 recorded |
| decoding | greedy (`do_sample=False`), **`repetition_penalty=1.0` passed explicitly**, `max_new_tokens=768` (EvalPlus default), batch size and left padding fixed |
| stop strings | closing fence, `\nif __name__`, `\ndef main(`, `\nprint(`, `<\|im_end\|>`, `<\|endoftext\|>` (EvalPlus's chat list) |
| sanitization | `evalplus.sanitize` |
| scoring | `evalplus.evaluate --dataset {humaneval,mbpp}`: base and plus tests |
| datasets | HumanEval+ v0.1.10, MBPP+ v0.2.0 (EvalPlus 0.3.1 defaults; pin the EvalPlus version) |
| quality | ruff `F,E9,B,SIM` (pinned version) and lizard max CC, on **passing** solutions only |
| **clean-pass@1** | passes base + plus tests **and** 0 ruff violations **and** max CC ≤ 10 |
| no-op rate | sanitized solution is all comments, or doesn't define the entry point |

**Isolated scoring environment.** EvalPlus pulls heavy dependencies (`openai`, `anthropic`, `google-generativeai`, `tree-sitter`), so it stays out of the project lock. Use `uvx --from evalplus==<pinned> evalplus.evaluate --dataset humaneval --samples run/humaneval-sanitized.jsonl`. `eval_results.json` holds the per-task base/plus status the paired tests need. EvalPlus recommends running evaluation inside Docker; v1's executor image can host it.

**Acceptance tests.** The module isn't used until all four pass:

| id | test | pass criterion |
|---|---|---|
| A1 | canonical solutions written as `samples.jsonl` | 100% base and plus on both datasets |
| A2 | bridge to v1.0: convert the five v1.0 `humaneval-chat` generation files to `samples.jsonl` and score the base tests | matches v1.0's pass@1 (90.2 / 70.7 / 53.0 / 45.1 / 24.4%) within 1–2 problems, each difference explained (EvalPlus compares outputs against the reference instead of running the original asserts) |
| A3 | `Qwen2.5-Coder-1.5B-Instruct` through `generate.py` + `score.py` | inside the 95% CI of its HumanEval / HumanEval+ / MBPP / MBPP+ numbers in the Qwen2.5-Coder technical report (Table 13) |
| A4 | the same run twice | identical generations; if bf16 kernels make greedy decoding nondeterministic, fix batch size and padding and document the residual rate |

### 2.6 Training runner (`benchmarks/code_sft/run.py`)

```
run.py experiments/<method>/<variant> [--lr X] [--seed S]
 1. load default_training_params.json, then shallow-merge training_params.json (PEFT semantics:
    an override of optimizer_kwargs replaces the whole dict, so repeat weight_decay)
 2. verify data hashes against manifest.json; verify the chat-template sha
 3. load base in bf16; get_peft_model(adapter_config.json); LoRA+ → peft.optimizers.create_loraplus_optimizer
 4. train: completion-only loss on the assistant turn; budget = max_steps × effective batch
 5. every eval_steps: valid-loss + valid-pass (greedy); keep the curve; no checkpoint selection on test
 6. final: test evaluation (§2.5) of the final step
 7. write results/<method>--<variant>[--lr..][--seed..].json (clean tree) or temporary_results/ (dirty tree)
```

`default_training_params.json` (starting point):

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-1.5B",
  "model_revision": "<pin>",
  "dtype": "bfloat16",
  "chat_template_sha256": "<pin>",
  "train_file": "benchmarks/data/train.jsonl",
  "max_seq_length": 1024,
  "batch_size": 8,
  "gradient_accumulation_steps": 2,
  "max_steps": 1000,
  "eval_steps": 200,
  "seed": 0,
  "grad_norm_clip": 1.0,
  "use_gc": true,
  "optimizer_type": "AdamW",
  "optimizer_kwargs": {"lr": 2e-4, "weight_decay": 0.0},
  "lr_scheduler": "cosine",
  "warmup_ratio": 0.03,
  "generation_kwargs": {"do_sample": false, "repetition_penalty": 1.0, "max_new_tokens": 768}
}
```

The budget is samples seen, not epochs: 1,000 steps × 16 = 16K samples ≈ 3 passes over 5K, identical for every method.

### 2.7 Methods: what to run first

Rank 16 where the method has a rank, on all linear projections (`q,k,v,o,gate,up,down`), so it is comparable with v1. The trainable-parameter count is always reported, which makes parameter-matched views possible in the viewer.

**Phase A (v2.0):**
1. LoRA baseline × 3 seeds at the default LR. This gives the noise floor and the per-run cost, and every later budget is computed from it.
2. Tier 1, 3 LRs each, selected on `valid-pass`:

| method | PEFT config | starting LR grid |
|---|---|---|
| LoRA (baseline) | `LoraConfig(r=16, lora_alpha=32)` | 1e-4, 2e-4, 5e-4 |
| rsLoRA | `use_rslora=True` | 5e-5, 1e-4, 2e-4 |
| DoRA | `use_dora=True` | 1e-4, 2e-4, 5e-4 |
| PiSSA init | `init_lora_weights="pissa"` | 5e-5, 1e-4, 2e-4 |
| LoRA+ | LoRA + `create_loraplus_optimizer(loraplus_lr_ratio=16)` | 5e-5, 1e-4, 2e-4 |
| LoKr | `LoKrConfig(r=16)` | 2e-4, 5e-4, 1e-3 |
| VeRA | `VeraConfig(r=256)` | 1e-3, 4e-3, 1e-2 |
| IA³ | `IA3Config` | 1e-3, 3e-3, 1e-2 |

These grids are starting points. **Edge rule:** if the best LR is at either end of the grid, add one more point beyond it. In PEFT image-gen, HiRA went from 0.555 to 0.708 DINOv2 between lr=1e-4 and 5e-4.

3. The 3 best configs get 2 more seeds and are compared paired against the LoRA baseline.

**Phase B (v2.1)** is in §3.

### 2.8 Results JSON (one per run)

```json
{
  "run_info": {"method": "dora", "variant": "r16", "lr": 2e-4, "seed": 0, "git_sha": "...", "dirty": false,
               "created_at": "...", "gpu": "...", "torch": "...", "transformers": "...", "peft": "...",
               "evalplus": "...", "ruff": "..."},
  "config": {"adapter_config": {}, "training_params": {}},
  "data": {"manifest_sha256": "...", "train_samples": 5000, "prompt_mix": {"new_file": 0.6, "edit": 0.4}},
  "train": {"trainable_params": 0, "peak_memory_reserved_mb": 0, "train_time_s": 0, "adapter_size_mb": 0,
            "loss_curve": [], "valid_pass_curve": []},
  "eval": {
    "protocol_version": "2.0", "eval_config": {},
    "humaneval": {"pass@1_base": 0, "pass@1_plus": 0, "ci_plus": [0, 0], "per_task": "path/to/eval_results.json"},
    "mbpp":      {"pass@1_base": 0, "pass@1_plus": 0, "ci_plus": [0, 0], "per_task": "path/to/eval_results.json"},
    "clean_pass@1": {"humaneval": 0, "mbpp": 0},
    "quality_passing_only": {"mean_cc": 0, "max_cc": 0, "ruff_per_100_loc": 0},
    "noop_rate": 0, "parse_failure_rate": 0, "gen_tokens_per_s": 0
  }
}
```

### 2.9 Viewer (`benchmarks/app.py`)

- Sortable table and Pareto plots: clean-pass@1 against trainable params, peak memory and train time.
- Paired comparison of any two runs, with McNemar per dataset.
- It refuses to place runs with different `protocol_version` or `manifest_sha256` on the same axis.

### 2.10 Done when (tag v2.0)

- A1–A4 pass.
- LoRA × 3 seeds and the 8 Tier-1 methods × 3 LRs are in `results/`; the losing LRs are in `graveyard/`.
- The top 3 have 3 seeds and paired tests against LoRA.
- The viewer runs locally.
- The README gets a "Results (v2.0)" section; `benchmarks/README.md` explains how to run and how to add an experiment.

---

## 3. v2.1: Stage 1 complete

- **Tier 2: the methods you already ran in PEFT's image-gen benchmark.** BD-LoRA, AdaMSS, HiRA, UniLoRA, BEFT, VeLoRA, TinyLoRA, Super-Tuning, OFT, MonteCLoRA (memory-hungry: it needed r=16 / `num_samples=2` on 16 GB for FLUX).
  - Analysis: Spearman rank correlation between each method's image-gen DINOv2 rank (PEFT's published results) and its code clean-pass@1 rank. Does a PEFT ranking transfer across modalities?
- **Tier 3: the rest of the catalogue.** AdaLoRA, LoHa, FourierFT, VB-LoRA, RandLoRA, SHiRA, C3A, MiSS, DeLoRA, WaveFT, GraLoRA, RoAd, Lily, PVeRA, PSOFT, OSF, FRoD, GLoRA, DEFT, Peanut. Also the LoRA initialisations OLoRA, EVA, CorDA and LoRA-GA, plus LoRA-FA. Lower bounds: prompt tuning, prefix tuning, LN tuning.
- **Hold:** HRA, until PEFT's planned refactor; X-LoRA and Poly, which need multiple adapters and are out of scope for single-task SFT.
- **Reference:** full fine-tuning, if it fits (8-bit AdamW + gradient checkpointing); otherwise state that it doesn't.
- **Data-size axis** for the top 5: 1K / 2.5K / 5K samples, same steps. Which method is most data-efficient?
- **Prompt-mix ablation** on LoRA and the winner: `new_file`/`edit` at 100/0, 60/40, 15/85 (≈ v1's mix). It measures how much the edit-heavy data of v1 matters on write-from-spec benchmarks.
- **Upstream:** open a Discussion on `huggingface/peft` proposing a code-generation task for `method_comparison/`, with the Tier-1 results attached. Reference the Discussion in any PR, or the triage bot may close it.

**Done when:** Tiers 2–3 are run or explicitly skipped with a reason, both axes are in the viewer, and the transfer analysis is written up.

---

## 4. v2.2: Stage 2, alignment with test-grounded preferences (`benchmarks/code_align/`)

### 4.1 Inputs

- **Reference policies:** the top 3 Stage-1 configs, each adapter merged into the bf16 base. Merging is lossless in bf16, unlike v1's 4-bit merge.
- **Prompt pool:** problems that come with tests. For every candidate source, verify the license and run the same 13-gram decontamination against HumanEval+ / MBPP+ / MBPP validation:
  - MBPP `train` + `prompt` splits, minus anything overlapping MBPP+;
  - APPS introductory (stdin/stdout tests, which need a small I/O harness);
  - KodCode or similar synthetic sets with tests;
  - SelfCodeAlign-style synthetic problems: the teacher writes problem + tests, and only tests its own solution passes are kept.

### 4.2 Sandbox

Reuse v1's Rust gRPC daemon unchanged. A candidate is sent as `solution + "\n\n" + tests`; exit code 0 means pass. For a pass fraction on sources whose tests are top-level `assert`s (MBPP, KodCode-style), wrap each one in a counter (tested: a solution failing 1 of 3 asserts prints `__RESULT__ 2/3`):

```python
import ast

def wrap_tests(tests: str) -> str:
    """Count passed top-level asserts instead of stopping at the first failure."""
    body = []
    for node in ast.parse(tests).body:
        if isinstance(node, ast.Assert):
            body.append(f"try:\n    {ast.unparse(node)}\n    __passed += 1\nexcept Exception:\n    pass\n__total += 1")
        else:
            body.append(ast.unparse(node))
    return "__passed = __total = 0\n" + "\n".join(body) + '\nprint(f"__RESULT__ {__passed}/{__total}")'
```

Raise the in-container timeout from 5 s to the dataset's own limit, so slow but correct solutions aren't counted as failures.

### 4.3 Pair construction: correctness first

- **Sampling:** for every prompt, sample N=8 candidates from each reference policy (T=0.8, top_p=0.95). The pairs are on-policy, per policy.
- **Pre-filter:** sanitize with EvalPlus's rules; drop no-op candidates (`noop.py`, from v1.1); run the tests.

| case | situation | label |
|---|---|---|
| A | some candidates pass all tests, some don't | chosen = a passing one, rejected = a failing one |
| C | ≥ 2 candidates pass | chosen = better quality (lower max CC, fewer ruff violations); **discard if equal** |
| B | none pass | **discarded**. A variant ranks them by pass fraction, *never* by quality alone (v1.0 finding 5). |

The reward ablation is done right this time:
- **execution-only** = Case A pairs; **composite** = Case A + Case C;
- both are drawn from the same prompts and subsampled to the **same pair count**;
- both are trained for the **same number of steps**.

### 4.4 Objectives (TRL 1.9.x; verify against your installed version)

| family | objective | how |
|---|---|---|
| offline pairwise | DPO | `DPOTrainer`, `loss_type=["sigmoid"]` |
| | IPO | `loss_type=["ipo"]` |
| | robust DPO | `loss_type=["robust"]`, `label_smoothing>0` |
| | APO-zero | `loss_type=["apo_zero"]` |
| | DPO + NLL (RPO-style) | `loss_type=["sigmoid","sft"]` |
| unpaired | KTO | `KTOTrainer` (now in `trl.trainer`); Case A passing = desirable, failing = undesirable |
| reference-free | ORPO | `trl.experimental.orpo` |
| | SimPO | `trl.experimental.cpo`, `loss_type="simpo"` |

The same pairs and the same steps go to every objective. LR is swept over {5e-6, 5e-5, 2e-4} at 1 epoch on `valid-pass`, and the winner is re-checked at 2 epochs. TRL's DPO/KTO default (1e-6) is for full fine-tuning; with LoRA r = 16, v1.1 found 5e-6 × 1 epoch barely moved the policy, 2e-4 × 1 epoch gave its best run and 2e-4 × 3 epochs collapsed.

**Diagnostics logged per run** (v1.0's lesson): reward accuracy and margins, the chosen-sequence log-probability (if it falls while the margin grows, the policy is pushing probability away from good code too), and the no-op rate on `valid-pass` every `eval_steps`.

**Done when:** every objective × 3 policies is in `results/`, the matched reward ablation is reported with paired tests, and each run's Δ against its own SFT policy has a CI.

---

## 5. v2.3: online RL and 7B re-validation

### 5.1 Online RL (`GRPOTrainer`, `RLOOTrainer`)

The reward function uses the same sandbox and the same no-op guard, and quality only counts once the tests pass. In the sketch below, `sanitize`, `is_noop`, `run_tests`, `max_cc` and `ruff_violations` are the v2 evaluation helpers; TRL passes the dataset columns `tests` and `entry_point` as keyword arguments:

```python
def code_reward(prompts, completions, tests, entry_point, **kwargs) -> list[float]:
    rewards = []
    for completion, t, ep in zip(completions, tests, entry_point):
        code = sanitize(completion, ep)
        if is_noop(code, ep):
            rewards.append(-1.0)
            continue
        passed, total = run_tests(code, t)            # Rust daemon
        r = passed / total
        if passed == total:                           # quality only after correctness
            r += 0.1 * (max_cc(code) <= 10) + 0.1 * (ruff_violations(code) == 0)
        rewards.append(r)
    return rewards
```

- Same prompt pool as §4.1, from the best Stage-2 policy and from its SFT policy.
- **Measure seconds per step first** (v1 measured 67 s/step at 7B). Use vLLM colocate if it fits next to training on 16 GB.
- **Comparison at an equal sample budget:** count the completions generated by RL and give the best offline objective the same number of sampled candidates when it builds its pairs.

### 5.2 7B re-validation

Run the top 3 end-to-end recipes (Stage-1 method + Stage-2 objective) on `Qwen2.5-Coder-7B` base with QLoRA (the v1 hardware setup), evaluated with protocol 2.0. This closes the loop with v1: same scale, a corrected recipe and corrected rewards.

### 5.3 Optional: multilingual evaluation

Use MultiPL-E HumanEval in cpp, cs, go, java, js, rs and ts, with a chat-prompt version of the MultiPL-E prompts. The stripped-prompt tokenization issue found in v1.0 applies to MultiPL-E's completion prompts too. Executors come from v1's Docker image, which already bundles all eight toolchains.

**Done when:** RL vs. offline at an equal budget is reported, the 7B re-validation table is in the README, and the s/step of each RL run is in its results JSON.

---

## 6. v3.0: distillation, quantization and serving (CodeAlign-Distill + CodeAlign-Runtime)

### 6.1 Setup

- **Teacher:** `Qwen2.5-Coder-7B-Instruct` (90.2% HumanEval under v1.0's protocol), or v2.3's best 7B recipe if it beats it on clean-pass@1. It is *not* v1's DPO models.
- **Students:** `Qwen2.5-Coder-1.5B` and `-0.5B`. They share the teacher's tokenizer, so logit-level KD needs no vocabulary mapping and TRL's GOLD (cross-tokenizer) isn't needed.
- **Student adapter:** the v2.1 winning PEFT config; full fine-tuning as a reference if memory allows.
- **Memory plan on 16 GB:** teacher in NF4 (~5 GB) next to a 1.5B bf16 student with an adapter for on-policy methods. Offline methods precompute teacher outputs, so the teacher is never resident during training.

### 6.2 Methods (`benchmarks/code_distill/`)

| method | data | teacher signal | tooling |
|---|---|---|---|
| SeqKD + rejection sampling | teacher samples on the §4.1 prompts, kept only if they pass the tests **and** are clean | tokens | SFT, `run.py` |
| logit KD (offline) | the same samples | teacher **top-k = 20 logprobs per token**, cached to disk (~120 bytes/token, so 8M tokens ≈ 1 GB) | custom loss (forward KL on the top-k support) |
| on-policy distillation | student samples | teacher logprobs on the student's tokens, reverse KL | `trl.experimental` `GKDTrainer` (`lmbda=1`: fully on-policy; `beta=1`: reverse KL) or `DistillationTrainer` (`beta=1`) |
| MiniLLM | student samples | reverse-KL policy gradient | `trl.experimental.minillm` |
| self-distillation (no teacher) | student samples, unverified | none | `trl.experimental` `SSDTrainer` |
| **specialists → generalist** (capstone) | per-specialist prompts | each prompt's specialist teacher | custom trainer |

**Capstone, modelled on DeepSeek-V4's post-training** (domain specialists trained with SFT + GRPO, merged into one model by on-policy distillation). Train two 1.5B specialists:
- **correctness:** v2.3's RL reward with the quality terms set to 0;
- **quality:** DPO on Case C pairs only.

Then distil both into one student by on-policy distillation, routing each prompt to its specialist. Both specialists and the student are adapters on the same frozen base: switch them with PEFT's `set_adapter`, so all three fit in memory at once. This is the custom part, because TRL's trainers expect a separate teacher model.

**Metrics:** everything in §2.8, plus the **teacher gap closed** = (student − student_init) / (teacher − student_init), and the teacher compute used (generated tokens, GPU-hours).

### 6.3 Quantization-aware training (the CodeAlign-Distill core)

1. **PTQ baselines** on the best distilled student: INT4 weight-only and NF4.
2. **QAT:** fine-tune with fake-quantized INT4 weights in the forward pass, using torchao's QAT flow (prepare → train → convert; pin the version and check its current API). The loss is the KD loss from §6.2: QAT + KD.
3. **Match the serving format.** The fake-quant scheme (group size, symmetric vs. asymmetric, scale dtype) must be the one CodeAlign-Runtime's fused INT4 kernel reads, as specified in that repo. A QAT model trained for a different layout loses the benefit at export.
4. Compare PTQ, QAT and QAT + KD on clean-pass@1 and the no-op rate.

### 6.4 Serving (CodeAlign-Runtime)

- Add a `runtime` backend to `benchmarks/evaluation/generate.py` that calls the Runtime (CLI or server) with the same prompts and stop strings. The INT4 kernels are then evaluated by the same protocol 2.0, which proves the serving path preserves quality.
- Report tokens/s, time to first token, p50/p95 latency and peak memory, against llama.cpp at the same quantization.
- Final artifact: a quality–latency Pareto plot covering 7B teacher → 1.5B / 0.5B students → PTQ / QAT → served by CodeAlign-Runtime.

**Done when:** each distillation method has a results JSON under protocol 2.0, QAT vs. PTQ is reported, and the served INT4 student's HumanEval+ / MBPP+ numbers come from the Runtime backend.

---

## 7. Repository, releases and compute

### 7.1 Layout

```
codealign/
├── src/                      # v1.x pipeline (unchanged except the v1.1 additions)
├── benchmarks/
│   ├── README.md             # how to run / add an experiment (written with v2.0)
│   ├── data/                 # build_slice.py, manifest.json
│   ├── evaluation/           # protocol 2.0 (§2.5), shared by every stage
│   ├── code_sft/             # Stage 1: default_training_params.json, experiments/, run.py, results/, temporary_results/, graveyard/
│   ├── code_align/           # Stage 2 + online RL
│   ├── code_distill/         # Stage 3 + QAT
│   └── app.py                # viewer
├── ROADMAP.md                # this file
└── README.md                 # one "Results (vX.Y)" section per release
```

### 7.2 Per release

- **Git:** tag + release notes (summary table + findings, same format as v1.0).
- **README:** a new results section; earlier ones stay unchanged.
- **Hugging Face Hub:**
  - the v2 data slice, as a new config or revision of the v1 dataset repo;
  - a `codealign-bench-results` dataset holding every results JSON and the graveyard;
  - the best adapters, with model cards that state measured scores and protocol version.

### 7.3 Compute: measure, then budget

Nothing here is measured yet. The LoRA baseline (§2.7, step 1) gives `T_train` and `T_eval` per run, and every budget follows from them:

```
Stage 1 Phase A ≈ (8 methods × 3 LRs) × (T_train + 5 × T_valid) + 3 winners × 2 seeds × T_train + finals × T_eval
```

For example, if `T_train` = 1 h and the valid/test evaluations take ~10 min, Phase A is roughly 30–35 GPU-hours. This is an illustration, not a measurement.

### 7.4 Risks

| risk | mitigation |
|---|---|
| A3 fails (published numbers not reproduced) | Stop. Diff prompts and stop strings against EvalPlus's `codegen` on the same model before training anything. |
| Greedy decoding nondeterministic across batch sizes | Fix batch size and padding; report the flip rate between two identical runs (A4). |
| A method breaks with bf16 or gradient checkpointing | Run it without GC as a separate variant; note it in the results JSON. |
| MonteCLoRA-style OOM | Reduce rank or samples and add a rank-matched LoRA baseline (as done for PEFT image-gen). |
| Prompt-pool license unclear | Exclude the source and keep the reason in `manifest.json`. |
| On-policy distillation too slow on 16 GB | Fall back to the offline methods (SeqKD, cached logit KD) and report the measured s/step. |

---

## 8. Upstream contributions along the way

- **huggingface/peft:** a code-generation task for `method_comparison/` (v2.1), plus each new method's code result as a data point for the image-gen ↔ code transfer question.
- **bigcode-evaluation-harness:** v1's patches are upstreamable:
  - transformers ≥ 5 4-bit loading;
  - the missing commas in `tasks/multiple.py`;
  - namespaced dataset ids;
  - an explicit `--repetition_penalty`.
  
  Two issues are worth reporting: the stripped-prompt tokenization problem with Qwen's tokenizer, and that the `humaneval` task isn't suited to chat models. Search for existing issues and PRs first.

---

## 9. Start here

1. [x] `scripts/audit_preference_pairs.py` and its table (v1.1 step 0, no GPU).
2. [x] `dpo_trainer.py` CLI overrides + `scripts/eval_checkpoint.sh`; run `v11-exec-lr5e-6-e1` and evaluate it.
3. [x] The rest of the v1.1 runs, `per_problem.py`, notebook 06, README, tag v1.1.
4. [ ] `benchmarks/evaluation/` + acceptance tests A1–A4. Nothing in v2 is trained before these pass.
5. [ ] `benchmarks/data/build_slice.py` + `manifest.json`.
6. [ ] `benchmarks/code_sft/run.py` from PEFT's MetaMathQA skeleton; LoRA baseline × 3 seeds → measured `T_train`, budget for Phase A.
7. [ ] Tier 1 × 3 LRs; viewer; seeds for the top 3; tag v2.0.
