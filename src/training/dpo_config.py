from src.data_curation.curation_config import BASE_DIR

# DPO MODELS
BASE_SFT_MODEL = str(BASE_DIR / "checkpoints" / "sft" / "merged_model")
TOKENIZER_MODEL = BASE_SFT_MODEL

# DPO PATHS
DPO_DS = str(BASE_DIR / "data" / "preferences" / "dpo_dataset.jsonl")
DPO_OUTPUT_DIR = BASE_DIR / "checkpoints" / "dpo"

# DPO HYPERPARAMETERS
BETA = 0.1
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 3 
MAX_SEQ_LENGTH = 2048
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
SEED = 42

# WANDB CONFIG
WANDB_RUN_NAME = "dpo-qwen2.5-coder-7b-composite-reward"