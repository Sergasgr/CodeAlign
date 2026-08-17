#!/usr/bin/env bash
# Phase 5 — Evaluation (Base vs. SFT vs. DPO Composite vs. DPO Ablation)
# Prerequisites:
#   git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git tools/bigcode-evaluation-harness
#   cd tools/bigcode-evaluation-harness && pip install -e .
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/evaluation

# Language to evaluate (default: python / humaneval)
LANG="${1:-python}"
echo "Evaluating language: ${LANG}"

if [ "$LANG" = "python" ]; then
  TASK="humaneval"
else
  TASK="multiple-${LANG}"
fi

# BASE MODEL
echo "═══ [1/4] Evaluating: Base Model ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --tasks "$TASK" \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/base_generations_${LANG}.json" \
  --metric_output_path "data/evaluation/base_metrics_${LANG}.json"

# SFT
echo "═══ [2/4] Evaluating: SFT Model ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/sft/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/sft_generations_${LANG}.json" \
  --metric_output_path "data/evaluation/sft_metrics_${LANG}.json"

# DPO — COMPOSITE REWARD
echo "═══ [3/4] Evaluating: DPO Composite Reward ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/dpo_composite_generations_${LANG}.json" \
  --metric_output_path "data/evaluation/dpo_composite_metrics_${LANG}.json"

# DPO - ABLATION
echo "═══ [4/4] Evaluating: DPO Ablation (execution-only) ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo_ablation/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/dpo_ablation_generations_${LANG}.json" \
  --metric_output_path "data/evaluation/dpo_ablation_metrics_${LANG}.json"

echo "═══ Computing static analysis (CC + lint) on all generations ═══"
uv run python -m src.evaluation.static_analysis --language "$LANG"

echo "═══ Extracting qualitative samples ═══"
uv run python -m src.evaluation.extract_qualitative --language "$LANG"

echo ""
echo "✓ Phase 5 evaluation complete for ${LANG}."
echo "  Metrics:      data/evaluation/*_metrics_${LANG}.json"
echo "  Static:       data/evaluation/static_analysis_results_${LANG}.jsonl"
echo "  Qualitative:  data/evaluation/qualitative_samples_${LANG}.md"
echo "  → Open notebooks/05_evaluation_report.ipynb for the full analysis."