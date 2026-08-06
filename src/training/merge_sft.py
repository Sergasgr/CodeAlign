import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.data_curation.curation_config import BASE_DIR
from src.training.sft_config import SFT_MODEL as BASE_MODEL_ID # "Qwen/Qwen2.5-Coder-7B-Instruct"
from src.training.dpo_config import BASE_SFT_MODEL as SFT_ADAPTER_DIR

def merge_and_save():
    MERGED_OUTPUT_DIR = str(BASE_DIR / "checkpoints" / "sft" / "merged_model")

    print(f"1. Loading base model ({BASE_MODEL_ID}) in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"2. Attaching SFT adapter from {SFT_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)

    print("3. Merging weights (merge_and_unload)...")
    model = model.merge_and_unload() # type: ignore

    print(f"4. Saving full merged model to {MERGED_OUTPUT_DIR}...")
    model.save_pretrained(MERGED_OUTPUT_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("Merge completed successfully!")

if __name__ == "__main__":
    merge_and_save()