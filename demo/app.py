import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data_curation.validators import linter_check

from demo.demo_config import (
    BASE_MODEL,
    SFT_CHECKPOINT,
    MERGED_SFT_MODEL,
    DPO_CHECKPOINT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    SERVER_PORT,
)

# ¿Va a ir cargando todos los modelos? ¿No debería hacer? -> 
# model.disable_adapters()
# model.set_adapter("sft")
# model.set_adapter("dpo")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=True,
)

print("Loading SFT adapter...")
sft_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=bnb_config,
        device_map="auto", 
        use_cache=True,
        adapter_name="sft"
    ),
    SFT_CHECKPOINT,
)

#print??
merged_sft_model = AutoModelForCausalLM.from_pretrained(
    MERGED_SFT_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading DPO adapter...")
dpo_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        MERGED_SFT_MODEL, 
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True,
        adapter_name="dpo"
    ),
    DPO_CHECKPOINT,
)

MODELS = {
    "Base": base_model,
    "SFT": sft_model,
    "DPO (composite)": dpo_model,
}

print("All models loaded.")

def generate(model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def format_metrics(code: str) -> str:
    result = linter_check(code, "python")
    cc = result.get("complexity")
    lint = result.get("lint_errors")
    cc_display = cc if cc is not None else "N/A (Syntax Error)"
    lint_display = lint if lint is not None else "N/A"
    return f"Cyclomatic complexity: {cc_display}  |  Lint errors: {lint_display}"

def compare(prompt: str):
    results = {}
    for name, model in MODELS.items():
        code = generate(model, prompt)
        metrics = format_metrics(code)
        results[name] = (code, metrics)

    return (
        results["Base"][0], results["Base"][1],
        results["SFT"][0], results["SFT"][1],
        results["DPO (composite)"][0], results["DPO (composite)"][1],
    )

EXAMPLES = [
    "Write a Python function that checks if a string is a valid palindrome, considering only alphanumeric characters and ignoring cases.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "Write a Python function that implements binary search on a sorted list and returns the index of the target element, or -1 if not found.",
    "Write a Python function to flatten a nested list of arbitrary depth.",
    "Write a Python class that implements a thread-safe LRU cache with get and put operations.",
]

with gr.Blocks(
    title="CodeAlign — Base vs. SFT vs. DPO",
    theme=gr.themes.Soft(), # type: ignore
) as demo:
    gr.Markdown(
        "# 🔬 CodeAlign — Side-by-Side Model Comparison\n"
        "Paste a coding prompt and compare **Base**, **SFT**, and **DPO (composite reward)** outputs.\n"
        "Each column shows the generated code and its static-analysis metrics "
        "(cyclomatic complexity + lint errors)."
    )

    prompt_input = gr.Textbox(
        label="Coding Prompt",
        placeholder="e.g. Write a Python function that...",
        lines=3,
    )
    run_btn = gr.Button("🚀 Generate & Compare", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Model")
            base_code = gr.Code(label="Base — Code", language="python")
            base_metrics = gr.Textbox(label="Base — Metrics", interactive=False)

        with gr.Column():
            gr.Markdown("### SFT Model")
            sft_code = gr.Code(label="SFT — Code", language="python")
            sft_metrics = gr.Textbox(label="SFT — Metrics", interactive=False)

        with gr.Column():
            gr.Markdown("### DPO (Composite Reward)")
            dpo_code = gr.Code(label="DPO — Code", language="python")
            dpo_metrics = gr.Textbox(label="DPO — Metrics", interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=prompt_input)

    run_btn.click( # type: ignore
        fn=compare,
        inputs=prompt_input,
        outputs=[base_code, base_metrics, sft_code, sft_metrics, dpo_code, dpo_metrics],
    )

if __name__ == "__main__":
    demo.launch(server_port=SERVER_PORT)