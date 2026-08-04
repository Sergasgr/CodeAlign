import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quant_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/sft/final_model", 
    adapter_name="sft"
)

model.load_adapter(
    "checkpoints/dpo/final_model", 
    adapter_name="dpo"
)