from datasets import concatenate_datasets, load_dataset
from src.data_curation.curation_config import LANGUAGE_CONFIGS, DS_NAME

def load_languages():
    parts = []
    for internal_name, config_name in LANGUAGE_CONFIGS.items():
        parquet_url = f"hf://datasets/{DS_NAME}@~parquet/{config_name}/train/**/*.parquet"
        ds = load_dataset("parquet", data_files=parquet_url, split="train")
        print(f"{internal_name} ({config_name!r}): {len(ds)} rows")
        if len(ds) == 0:
            print(f"WARNING: 0 rows for {internal_name} — check this config name against the dataset's actual config list")
        ds = ds.add_column("target_language", [internal_name] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)