from pathlib import Path

# DATA CURATION PATHS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CURATED_DIR = DATA_DIR / "curated"
DS_PATH = CURATED_DIR / "pristine_dataset.jsonl"
REJECTED_PATH = CURATED_DIR / "rejected_dataset.jsonl"
REPORT_PATH = CURATED_DIR / "report_dataset.jsonl"

# CommitPackFT
DS_NAME = "bigcode/commitpackft" #"ise-uiuc/Magicoder-OSS-Instruct-75K"
LANGUAGE_CONFIGS = {
    "python": "python",
    "java": "java",
    "cpp": "c++",
    "c_sharp": "c#",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust"
}

# ALLOWED LICENSES
ALLOWED_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "isc", "unlicense", "cc0-1.0",
}

# DATA CURATION PARAMETERS 
MAX_LINT_ERRORS = 3
MAX_COMPLEXITY = 10
THRESHOLD = 0.85

# MINHAS/LSH PARAMETERS
DEDUP_THRESHOLD = 0.85
MINHASH_NUM_PERM = 128

# CODE SMELLS PARAMETERS
DUPLICATION_WINDOW = 12
DUPLICATION_DENSITY_THRESHOLD = 0.2

NEW_FILE_LINE_THRESHOLD = 3