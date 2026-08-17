import json
import os
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

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

LANGUAGE_LOGS = { 
    "python": "Python",
    "java": "Java",
    "cpp": "C++",
    "c_sharp": "C#",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust"
}

language_count: dict[str, int] = {}

def process_sample_heavy(sample):
    prompt, prompt_type = build_prompt(sample)
    language = sample["target_language"]
    code_data = check_code(sample["new_contents"], language)
    return sample, prompt, prompt_type, language, code_data

def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    if DS_PATH.exists():
        print(f"{DS_PATH} already exists — delete it if you want to regenerate.")
        return

    print("Loading CommitPackFT (one load per language config)...")
    ds = load_languages()
    total_samples = len(ds)
    
    ds = ds.filter(lambda row: (row["license"] or "").lower() in ALLOWED_LICENSES)
    print(f"After license filter ({sorted(ALLOWED_LICENSES)}): {len(ds)} rows")
    
    total_lang_counts = Counter(ds["target_language"])
    
    dup_index = DuplicateIndex(threshold=DEDUP_THRESHOLD)
    counts: dict[str, int] = {}
    prompt_type_counts: dict[str, int] = {}
    processed_lang_counts: dict[str, int] = {}
    
    max_workers = os.cpu_count()
    print(f"\nStarting parallel generation with {max_workers} threads...\n", flush=True)

    with open(DS_PATH, 'w', encoding='utf-8') as f, \
         open(REJECTED_PATH, 'w', encoding='utf-8') as rj, \
         open(REPORT_PATH, 'w', encoding='utf-8') as rp, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            futures = [executor.submit(process_sample_heavy, sample) for sample in ds]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                sample, prompt, prompt_type, language, code_data = future.result()
                processed_lang_counts[language] = processed_lang_counts.get(language, 0) + 1
                
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
                    language_count[language] = language_count.get(language, 0) + 1
                    prompt_type_counts[prompt_type] = prompt_type_counts.get(prompt_type, 0) + 1
                    f.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                else:
                    rj.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                
                if i % 1000 == 0: # Ahora que lo lanzo en paralelo y va más ráido quizás podría considerar subir de 1000
                    accepted_so_far = counts.get("accepted", 0)
                    progress_pct = (i / len(ds)) * 100
                    acceptance_rate = (accepted_so_far / i * 100) if i > 0 else 0
                    
                    lang_lines = []
                    for lang_key in total_lang_counts.keys():
                        processed_count = processed_lang_counts.get(lang_key, 0)
                        total_lang = total_lang_counts[lang_key]
                        pct_lang = (processed_count / total_lang * 100) if total_lang > 0 else 0
                        lang_lines.append(f"- {LANGUAGE_LOGS[lang_key]}: {processed_count}/{total_lang} ({pct_lang:.2f}%)")
                    
                    print(
                      f"\rProcessing... {i}/{len(ds)} ({progress_pct:.2f}%) | "
                      f"Accepted: {accepted_so_far} ({acceptance_rate:.2f}%)", 
                      flush=True
                    )
                
                    print(f"Languages Progress:\n" + "\n".join(lang_lines) + "\n", flush=True)
                                  
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
    print("\nAccepted samples by language:")
    for lang, n in sorted(language_count.items(), key=lambda kv: -kv[1]):
        print(f"{LANGUAGE_LOGS[lang]}: {n}")
    print(f"Full report: {REPORT_PATH}")
                  
if __name__ == "__main__":
    main()