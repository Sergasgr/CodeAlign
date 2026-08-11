#!/usr/bin/env bash
# Prerequisites:
#   git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git tools/bigcode-evaluation-harness
#   cd tools/bigcode-evaluation-harness && pip install -e .

set -euo pipefail
mkdir -p data/evaluation

# BASE MODEL
echo "═══ [1/4] Evaluating: Base Model ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/base_generations.json \
  --metric_output_path data/evaluation/base_metrics.json

# SFT
echo "═══ [2/4] Evaluating: SFT Model ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/sft/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/sft_generations.json \
  --metric_output_path data/evaluation/sft_metrics.json

# DPO — COMPOSITE REWARD
echo "═══ [3/4] Evaluating: DPO Composite Reward ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/dpo_composite_generations.json \
  --metric_output_path data/evaluation/dpo_composite_metrics.json

# DPO - ABLATION
echo "═══ [4/4] Evaluating: DPO Ablation (execution-only) ═══"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo_ablation/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/dpo_ablation_generations.json \
  --metric_output_path data/evaluation/dpo_ablation_metrics.json

echo "═══ Computing static analysis (CC + lint) on all generations ═══"
uv run python -m src.evaluation.metrics_analyzer

echo "═══ Extracting qualitative samples ═══"
uv run python -m src.evaluation.extract_qualitative

echo ""
echo "✓ Phase 5 evaluation complete."
echo "  Metrics: data/evaluation/*_metrics.json"
echo "  Static: data/evaluation/static_analysis_results.jsonl"
echo "  Qualitative: data/evaluation/qualitative_samples.md"
echo "  → Open notebooks/05_evaluation_report.ipynb for the full analysis."