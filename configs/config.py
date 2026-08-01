"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

DS_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
DS_PATH = os.path.join(DATASETS_DIR, "pristine_dataset.jsonl")
REJECTED_PATH = os.path.join(DATASETS_DIR, "rejected_dataset.jsonl")
REPORT_PATH = os.path.join(DATASETS_DIR, "report_dataset.jsonl")
#corregir rutas teniendo en cuenta la division del proyecto en carpetas

MAX_LINT_ERRORS = 3
MAX_COMPLEXITY = 10
"""
from pathlib import Path

# DATA CURATION PATHS
BASE_DIR = Path(__file__).resolve().parents[2] # codealign/data -> curated/ preferences/ raw/ 
DATA_DIR = BASE_DIR / "data"
CURATED_DIR = DATA_DIR / "curated"
 
DS_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
DS_PATH = CURATED_DIR / "pristine_dataset.jsonl"
REJECTED_PATH = CURATED_DIR / "rejected_dataset.jsonl"
REPORT_PATH = CURATED_DIR / "report_dataset.jsonl"

# DATA CURATION PARAMETERS 
MAX_LINT_ERRORS = 3
MAX_COMPLEXITY = 10
THRESHOLD = 0.85

# SFT MODELS
TOKENIZER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct" 
SFT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# SFT PATHS
MODELS_DIR = BASE_DIR / "models"
SFT_OUTPUT_DIR = str(MODELS_DIR / "sft-checkpoint")

# SFT HYPERPARAMETERS 
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2

# WANDB CONFIG
WANDB_RUN_NAME = "sft-qwen-7b-pristine"