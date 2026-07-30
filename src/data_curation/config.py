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