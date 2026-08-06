from src.data_curation.curation_config import BASE_DIR

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SFT_CHECKPOINT = str(BASE_DIR / "checkpoints" / "sft" / "final_model")
MERGED_SFT_MODEL = str(BASE_DIR / "checkpoints" / "sft" / "merged_model") 
DPO_CHECKPOINT = str(BASE_DIR / "checkpoints" / "dpo" / "final_model")

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2          
TOP_P = 0.95
REPETITION_PENALTY = 1.1

SERVER_PORT = 7860