from src.data_curation.curation_config import BASE_DIR

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SFT_CHECKPOINT = str(BASE_DIR / "checkpoints" / "sft" / "final_model")
DPO_COMPOSITE_CHECKPOINT = str(BASE_DIR / "checkpoints" / "dpo" / "final_model")
DPO_ABLATION_CHECKPOINT = str(BASE_DIR / "checkpoints" / "dpo_ablation" / "final_model")

EVAL_DIR = BASE_DIR / "data" / "evaluation"

BASE_GENERATIONS = str(EVAL_DIR / "base_generations.json")
SFT_GENERATIONS = str(EVAL_DIR / "sft_generations.json")
DPO_COMPOSITE_GENERATIONS = str(EVAL_DIR / "dpo_composite_generations.json")
DPO_ABLATION_GENERATIONS = str(EVAL_DIR / "dpo_ablation_generations.json")

BASE_METRICS = str(EVAL_DIR / "base_metrics.json")
SFT_METRICS = str(EVAL_DIR / "sft_metrics.json")
DPO_COMPOSITE_METRICS = str(EVAL_DIR / "dpo_composite_metrics.json")
DPO_ABLATION_METRICS = str(EVAL_DIR / "dpo_ablation_metrics.json")

STATIC_ANALYSIS_RESULTS = str(EVAL_DIR / "static_analysis_results.jsonl")

QUALITATIVE_SAMPLES_MD = str(EVAL_DIR / "qualitative_samples.md")
TARGET_SAMPLES = 10
COMPLEXITY_GAP = 2  # minimum CC difference to qualify as interesting