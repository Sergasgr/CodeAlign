import json
import os
from datasets import load_dataset
from config import DS_NAME, DS_PATH, REJECTED_PATH, REPORT_PATH, DATASETS_DIR
from validators import parse_code, check_code

def dataset_generator(ds):
    for sample in ds["train"]:
        input_prompt = sample["problem"]
        output_code = sample["solution"]
        yield {"input_prompt": input_prompt, "output_code": output_code}

def to_chatml(reg, is_valid, lint_errors, complexity) -> json:
    return {
        "messages": [
            {"role": "user", "content": reg["input_prompt"]},
            {"role": "assistant", "content": reg["output_code"]}
        ],
        "metadata": {
            "is_valid_ast": is_valid,
            "lint_errors": lint_errors,
            "cyclomatic_complexity": complexity
        }
    }
        
def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    ds = load_dataset(DS_NAME)
    if not os.path.exists(DS_PATH):
        with open(DS_PATH, 'w', encoding='utf-8') as f, open(REJECTED_PATH, 'w', encoding='utf-8') as rj, open(REPORT_PATH, 'w', encoding='utf-8') as rp:
            for reg in dataset_generator(ds):
                language, parsed_code = parse_code(reg["output_code"])
                code_data = check_code(parsed_code, language)
                chat_ml = to_chatml(reg, code_data["is_valid_ast"], code_data["lint_errors"], code_data["cyclomatic_complexity"])
                if code_data["flag"]: 
                    f.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                else:
                    chat_ml["metadata"]["error"] = code_data["error"]
                    rj.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                chat_ml["metadata"]["status"] = code_data["status"]
                rp.write(json.dumps(chat_ml, ensure_ascii=False) + "\n")
                    
if __name__ == "__main__":
    main()