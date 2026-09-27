set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f tools/bigcode-evaluation-harness/main.py ]; then
  echo "Harness not found — run: bash scripts/setup_eval_harness.sh" >&2
  exit 1
fi

mkdir -p data/evaluation

EVAL_LANG="${1:-python}"
if [ "$EVAL_LANG" != "python" ]; then
  echo "Only 'python' (HumanEval) is supported in v1.0." >&2
  exit 1
fi
echo "Evaluating language: ${EVAL_LANG}"
TASK="humaneval-chat"
MAX_LENGTH=1536
PROMPTS="data/evaluation/humaneval_chat_prompts.json"

uv run python -m src.evaluation.humaneval_chat_prompts --out "$PROMPTS"

move_generations() {
  mv "data/evaluation/${1}_generations_${EVAL_LANG}_${TASK}.json" "data/evaluation/${1}_generations_${EVAL_LANG}.json"
}

# BASE MODEL
echo "═══ [1/4] Evaluating: Base Model ═══"
uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --load_in_4bit \
  --tasks "$TASK" \
  --load_data_path "$PROMPTS" \
  --max_length_generation "$MAX_LENGTH" \
  --precision bf16 \
  --do_sample False \
  --repetition_penalty 1.0 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/base_generations_${EVAL_LANG}.json" \
  --metric_output_path "data/evaluation/base_metrics_${EVAL_LANG}.json"
move_generations base

# SFT
echo "═══ [2/4] Evaluating: SFT Model ═══"
uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/sft/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --load_data_path "$PROMPTS" \
  --max_length_generation "$MAX_LENGTH" \
  --precision bf16 \
  --do_sample False \
  --repetition_penalty 1.0 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/sft_generations_${EVAL_LANG}.json" \
  --metric_output_path "data/evaluation/sft_metrics_${EVAL_LANG}.json"
move_generations sft

# DPO — COMPOSITE REWARD
echo "═══ [3/4] Evaluating: DPO Composite Reward ═══"
uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --load_data_path "$PROMPTS" \
  --max_length_generation "$MAX_LENGTH" \
  --precision bf16 \
  --do_sample False \
  --repetition_penalty 1.0 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/dpo_composite_generations_${EVAL_LANG}.json" \
  --metric_output_path "data/evaluation/dpo_composite_metrics_${EVAL_LANG}.json"
move_generations dpo_composite

# DPO - ABLATION
echo "═══ [4/4] Evaluating: DPO Ablation (execution-only) ═══"
uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model checkpoints/sft/merged_model \
  --peft_model checkpoints/dpo_ablation/final_model \
  --load_in_4bit \
  --tasks "$TASK" \
  --load_data_path "$PROMPTS" \
  --max_length_generation "$MAX_LENGTH" \
  --precision bf16 \
  --do_sample False \
  --repetition_penalty 1.0 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "data/evaluation/dpo_ablation_generations_${EVAL_LANG}.json" \
  --metric_output_path "data/evaluation/dpo_ablation_metrics_${EVAL_LANG}.json"
move_generations dpo_ablation

# DPO — COMPOSITE, SIZE-MATCHED
if [ -d checkpoints/dpo_composite_matched/final_model ]; then
  echo "═══ [5/5] Evaluating: DPO Composite, size-matched (control) ═══"
  uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
    --model checkpoints/sft/merged_model \
    --peft_model checkpoints/dpo_composite_matched/final_model \
    --load_in_4bit \
    --tasks "$TASK" \
    --load_data_path "$PROMPTS" \
    --max_length_generation "$MAX_LENGTH" \
    --precision bf16 \
    --do_sample False \
    --repetition_penalty 1.0 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "data/evaluation/dpo_composite_matched_generations_${EVAL_LANG}.json" \
    --metric_output_path "data/evaluation/dpo_composite_matched_metrics_${EVAL_LANG}.json"
  move_generations dpo_composite_matched
fi

echo "═══ Computing static analysis (CC + lint) on all generations ═══"
uv run python -m src.evaluation.static_analysis --language "$EVAL_LANG"

echo "═══ Extracting qualitative samples ═══"
uv run python -m src.evaluation.extract_qualitative --language "$EVAL_LANG"

echo ""
echo "✓ Phase 5 evaluation complete for ${EVAL_LANG}."
echo "  Metrics:      data/evaluation/*_metrics_${EVAL_LANG}.json"
echo "  Static:       data/evaluation/static_analysis_results_${EVAL_LANG}.jsonl"
echo "  Qualitative:  data/evaluation/qualitative_samples_${EVAL_LANG}.md"
echo "  → Open src/notebooks/05_evaluation_report.ipynb for the full analysis."
