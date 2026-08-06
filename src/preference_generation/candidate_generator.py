import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.preference_generation.preference_generation_config import (
    SFT_MODEL,
    TOKENIZER_MODEL_PREFERENCE_GENERATOR,
    TEMPERATURE,
    TOP_P,
    N_CANDIDATES
)
from src.data_curation.validators import parse_code

class CandidateGenerator():
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL,
            quantization_config=quantization_config,
            device_map="auto",                    
            use_cache=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL_PREFERENCE_GENERATOR,
            use_fast=True, 
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_candidates(self, prompt: str, n_candidates: int = N_CANDIDATES) -> list[str]:
        chat_prompt = self.tokenizer.apply_chat_template(
            {"role": "user", "content": prompt},
            tokenize=False,
            add_generation_prompt=True 
        )
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        outputs = self.model.generate( # type: ignore
            **inputs,
            do_sample=True,      
            temperature=TEMPERATURE,     
            top_p=TOP_P,      
            max_new_tokens=1024,
            num_return_sequences=n_candidates,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        input_length = inputs["input_ids"].shape[1]
        candidates = []
        
        for output in outputs:
            generated_tokens = output[input_length:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            clean_code = parse_code(raw_text)
            candidates.append(clean_code)
            
        return candidates
            