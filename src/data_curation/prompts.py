from src.data_curation.curation_config import NEW_FILE_LINE_THRESHOLD

def is_new_file(sample: dict) -> bool:
    return len(sample["old_contents"].strip().splitlines()) <= NEW_FILE_LINE_THRESHOLD

def choose_instruction(sample: dict) -> str:
    subject = sample["subject"].strip()
    message = sample["message"].strip()
    return message if len(message) > len(subject) else subject

def build_prompt(sample: dict) -> tuple[str, str]:
    instruction = choose_instruction(sample)
    if is_new_file(sample):
        prompt = f"Write {sample['target_language']} code for the following: {instruction}"
        return prompt, "new_file"
    prompt = (
        f"Here is the current file:\n```{sample['target_language']}\n{sample['old_contents']}\n```\n\n"
        f"Apply this change: {instruction}"
    )
    return prompt, "edit"

def to_chatml(sample: dict, prompt: str, prompt_type: str, code_data: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample["new_contents"]},
        ],
        "metadata": {
        "language": sample["target_language"],
        "license": sample["license"],
        "prompt_type": prompt_type,
        "is_valid_syntax": code_data["is_valid_syntax"],
        "lint_errors": code_data["lint_errors"],
        "cyclomatic_complexity": code_data["cyclomatic_complexity"],
        "status": code_data["status"],
        "error": code_data["error"],
        },
    }