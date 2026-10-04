set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <name> <base_model> <adapter_path>" >&2
    exit 1
fi

NAME=$1
BASE_MODEL=$2
ADAPTER=$3

if [ ! -f tools/bigcode-evaluation-harness/main.py ]; then
  echo "Harness not found — run: bash scripts/setup_eval_harness.sh" >&2
  exit 1
fi
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "No adapter at $ADAPTER (adapter_config.json missing) — has the training run finished?" >&2
  exit 1
fi

EVAL_LANG="python"
TASK="humaneval-chat"
MAX_LENGTH=1536
PROMPTS="data/evaluation/humaneval_chat_prompts.json"
OUTPUT_METRICS="data/evaluation/${NAME}_metrics_${EVAL_LANG}.json"
OUTPUT_GENS="data/evaluation/${NAME}_generations_${EVAL_LANG}.json"

mkdir -p data/evaluation

if [ ! -f "$PROMPTS" ]; then
  uv run python -m src.evaluation.humaneval_chat_prompts --out "$PROMPTS"
fi

echo "═══ Evaluating: $NAME ($ADAPTER on $BASE_MODEL) ═══"
uv run accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model "$BASE_MODEL" \
  --peft_model "$ADAPTER" \
  --load_in_4bit \
  --tasks "$TASK" \
  --load_data_path "$PROMPTS" \
  --max_length_generation "$MAX_LENGTH" \
  --precision bf16 \
  --do_sample False \
  --repetition_penalty 1.0 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "$OUTPUT_GENS" \
  --metric_output_path "$OUTPUT_METRICS"

mv "data/evaluation/${NAME}_generations_${EVAL_LANG}_${TASK}.json" "$OUTPUT_GENS"

echo "═══ Static analysis (CC + lint): $NAME ═══"
uv run python -m src.evaluation.static_analysis --language "$EVAL_LANG" --models "$NAME"

echo "Evaluation complete. Metrics saved to $OUTPUT_METRICS"