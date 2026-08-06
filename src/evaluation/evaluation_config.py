BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SFT_CHECKPOINT = "checkpoints/sft/final_model"
DPO_COMPOSITE_CHECKPOINT = "checkpoints/dpo/final_model"
DPO_ABLATION_CHECKPOINT = "checkpoints/dpo_ablation/final_model"

BASE_GENERATIONS = "data/evaluation/base_generations.json"
SFT_GENERATIONS = "data/evaluation/sft_generations.json"
DPO_COMPOSITE_GENERATIONS = "data/evaluation/dpo_composite_generations.json"
DPO_ABLATION_GENERATIONS = "data/evaluation/dpo_ablation_generations.json"

BASE_METRICS = "data/evaluation/base_metrics.json"
SFT_METRICS = "data/evaluation/sft_metrics.json"
DPO_COMPOSITE_METRICS = "data/evaluation/dpo_composite_metrics.json"
DPO_ABLATION_METRICS = "data/evaluation/dpo_ablation_metrics.json"

STATIC_ANALYSIS_RESULTS = "data/evaluation/static_analysis_results.jsonl"

QUALITATIVE_SAMPLES_MD = "data/evaluation/qualitative_samples.md"
TARGET_SAMPLES = 10
COMPLEXITY_GAP = 2  # minimum CC difference to qualify as interesting