import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.demo.models_initializer import base_model, model, tokenizer
from src.data_curation.validators import linter_check, parse_code
from src.demo.demo_config import TEMPERATURE_DEMO, TOP_P_DEMO
from src.demo.demo_entitites import GenerateRequest, ModelResult, ComparisonResponse

app = FastAPI(title="CodeAlign Inference API", version="1.0.0")
inference_lock = asyncio.Lock()

def execute_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        do_sample=True,      
        temperature=TEMPERATURE,
        top_p=TOP_P,      
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id
    )
    
    input_length = inputs["input_ids"].shape[1]
    raw_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return parse_code(raw_text)

def get_result(prompt: str) -> ModelResult:
    code = execute_model(prompt)
    metrics = linter_check(code, "python")
    return ModelResult(
        code=code,
        complexity=metrics.get("complexity") or 0,
        lint_errors=metrics.get("lint_errors") or 0
    )

@app.post("/generate_comparison", response_model=ComparisonResponse)
async def generate_comparison_endpoint(request: GenerateRequest):
    async with inference_lock:
        try:
            model.disable_adapters()
            base_res = get_result(request.prompt)
            
            model.set_adapter("sft")
            sft_res = get_result(request.prompt)
            
            model.set_adapter("dpo")
            dpo_res = get_result(request.prompt)
            
            return ComparisonResponse(base=base_res, sft=sft_res, dpo=dpo_res)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)