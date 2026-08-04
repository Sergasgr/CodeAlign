# PREFERENCE GENERATOR MODELS
SFT_MODEL = "checkpoints/sft/final_model"
TOKENIZER_MODEL_PREFERENCE_GENERATOR = "Qwen/Qwen2.5-Coder-7B-Instruct"

# PREFERENCE GENERATION - DPO DATASET PATH
DPO_DS = "data/preferences/dpo_dataset.jsonl"

# PREFERENCE GENERATION HYPERPARAMETERS
N_CANDIDATES = 2 # o 4??
TEMPERATURE = 0.7 # entre 0.6 y 0.8
TOP_P = 0.95

# SANDBOX CONFIG
SANDBOX_IMAGE_NAME = "codealign-executor:latest"
EXECUTION_TIMEOUT = 2

# COMPOSITE REWARD PARAMETERS
W_EXEC = 1.0 
W_COMPLEXITY = 0.1 
W_LINT = 0.2 

""" ¿Fase 5?
Paso 0: El Checkpoint de Ablación (El Prerrequisito)

Antes de evaluar, necesitas tener qué evaluar. Para que tu hipótesis tenga validez empírica, debes generar un modelo de control.
Lo que debes hacer: Modificar temporalmente tu preference_generation_config.py para poner W_COMPLEXITY = 0 y W_LINT = 0 y dejar solo W_EXEC = 1.0.
Ejecución: Correr tu pipeline de las fases 3 y 4 con esta configuración para obtener un modelo que llamaremos checkpoints/dpo_ablation/final_model.
"""

# PREFERENCE GENERATION REPORT
DPO_REPORT = "data/preferences/dpo_report.jsonl"

# WANDB CONFIG
WANDB_RUN_NAME = "phase3-preference-generation"