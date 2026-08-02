import json
from src.data_curation.curation_config import (
    ALLOWED_LICENSES,
    CURATED_DIR,
    DEDUP_THRESHOLD,
    DS_PATH,
    REJECTED_PATH,
    REPORT_PATH,
)
from src.data_curation.minhash import DuplicateIndex
from src.data_curation.validators import check_code
from src.data_curation.dataset import load_languages
from src.data_curation.prompts import build_prompt, to_chatml

# Run this as a module from the project root: `uv run python -m
# src.data_curation.main` (see scripts/01_curate_data.sh).

def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    if DS_PATH.exists():
        print(f"{DS_PATH} already exists — delete it if you want to regenerate.")
        return

    print("Loading CommitPackFT (one load per language config)...")
    ds = load_languages()
    
    ds = ds.filter(lambda row: (row["license"] or "").lower() in ALLOWED_LICENSES)
    print(f"After license filter ({sorted(ALLOWED_LICENSES)}): {len(ds)} rows")
    
    dup_index = DuplicateIndex(threshold=DEDUP_THRESHOLD)
    counts: dict[str, int] = {}
    prompt_type_counts: dict[str, int] = {}

    with open(DS_PATH, 'w', encoding='utf-8') as f, \
         open(REJECTED_PATH, 'w', encoding='utf-8') as rj, \
         open(REPORT_PATH, 'w', encoding='utf-8') as rp:
             
            for sample in ds:
                prompt, prompt_type = build_prompt(sample)
                code_data = check_code(sample["new_contents"], sample["target_language"])
    
                if code_data["flag"]:
                    dup_of = dup_index.check_and_add(sample["new_contents"])
                    if dup_of is not None:
                        code_data = {
                            **code_data,
                            "flag": False,
                            "status": "rejected_duplicate",
                            "error": f"Near-duplicate of sample {dup_of}",
                        }
                chat_ml = to_chatml(sample, prompt, prompt_type, code_data)
                rp.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                
                status = code_data["status"]
                counts[status] = counts.get(status, 0) + 1
                if code_data["flag"]: 
                    prompt_type_counts[prompt_type] = prompt_type_counts.get(prompt_type, 0) + 1
                    f.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                else:
                    rj.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                    
    total = sum(counts.values())
    accepted = counts.get("accepted", 0)
    print(f"\nDone: {accepted}/{total} accepted ({accepted / total:.1%})" if total else "Done: 0 rows processed")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{status}: {n}")
    print("\nPrompt format among accepted samples:")
    for ptype, n in sorted(prompt_type_counts.items(), key=lambda kv: -kv[1]):
        print(f"{ptype}: {n}")
    print(f"\nCurated dataset: {DS_PATH}")
    print(f"Rejected samples: {REJECTED_PATH}")
    print(f"Full report: {REPORT_PATH}")
                  
if __name__ == "__main__":
    main()