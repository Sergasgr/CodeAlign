import json
import os
from datasets import load_dataset
from configs.config import CURATED_DIR, DS_NAME, DS_PATH, REJECTED_PATH, REPORT_PATH #, THRESHOLD
from src.data_curation.validators import check_code, parse_code
#from src.data_curation.minhash import tokenize_code, generate_minhash, generate_minhash_lsh, duplicates_prevention, insert_minhash_to_lsh

#lsh = generate_minhash_lsh(THRESHOLD)

def dataset_generator(ds):
    for sample in ds["train"]:
        input_prompt = sample["problem"]
        output_code = sample["solution"]
        yield {"input_prompt": input_prompt, "output_code": output_code}
        
"""
def dataset_generator(ds):
    valid_langs = {"python", "java", "cpp", "c++", "c#", "javascript", "typescript"}
    valid_licenses = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}

    for sample in ds["train"]:
        lang = sample.get("lang", "").lower()
        license_type = sample.get("license", "").lower()

        if lang in valid_langs and license_type in valid_licenses:
            yield {
                "lang": lang,
                "subject": sample["subject"],
                "new_contents": sample["new_contents"]
            }
"""

def to_chatml(reg, is_valid, lint_errors, complexity) -> dict:
    return {
        "messages": [
            {"role": "user", "content": reg["input_prompt"]},
            {"role": "assistant", "content": reg["output_code"]},
        ],
        "metadata": {
            "is_valid_ast": is_valid,
            "lint_errors": lint_errors,
            "cyclomatic_complexity": complexity,
        },
    }
        
def main():
    os.makedirs(CURATED_DIR, exist_ok=True)
    ds = load_dataset(DS_NAME)
    if not os.path.exists(DS_PATH):
        with open(DS_PATH, 'w', encoding='utf-8') as f, \
             open(REJECTED_PATH, 'w', encoding='utf-8') as rj, \
             open(REPORT_PATH, 'w', encoding='utf-8') as rp:
            for reg in dataset_generator(ds):
                language, parsed_code = parse_code(reg["output_code"])
                """
                code_duplicated, mh = duplicates_prevention(parsed_code, lsh)
                if not code_duplicated:
                lsh = insert_minhash_to_lsh(lsh, mh)
                code_data = check_code(parsed_code, language, code_duplicated)
                """
                code_data = check_code(parsed_code, language)
                chat_ml = to_chatml(
                    reg, 
                    code_data.get("is_valid_ast", False), 
                    code_data.get("lint_errors", 0), 
                    code_data.get("cyclomatic_complexity", 0)
                )
                if code_data.get("flag"): 
                    f.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                else:
                    chat_ml["metadata"]["error"] = code_data["error"]
                    rj.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                chat_ml["metadata"]["status"] = code_data["status"]
                rp.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                    
if __name__ == "__main__":
    main()