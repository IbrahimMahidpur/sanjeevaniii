"""Merge LoRA adapter into base model.

Loads the base Mistral‑7B‑Instruct‑v0.3 model and the LoRA checkpoint produced by ``qlora_train.py``.
Merges the adapters (``model.merge_and_unload()``) and saves the resulting merged model to ``./merged_model``.
A simple sanity‑check inference is performed to confirm the model loads correctly.
"""

import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main() -> None:
    # Paths – expect the LoRA checkpoint under the output_dir defined in config.yaml
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in the current directory.")
    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_name = cfg["model_name"]
    checkpoint_dir = Path(cfg.get("output_dir", "./checkpoints"))
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_dir}")

    # 4‑bit NF4 quantisation – same as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Attach LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    # Merge adapters into the base model
    merged_model = model.merge_and_unload()
    # Save merged model
    out_dir = Path("merged_model")
    out_dir.mkdir(exist_ok=True)
    merged_model.save_pretrained(out_dir)
    # Also save tokenizer for completeness
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    # Simple sanity‑check: generate a short response to a dummy prompt
    tokenizer.pad_token = tokenizer.eos_token
    prompt = "[INST] Summarize the following medical abstract. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
    with torch.no_grad():
        output_ids = merged_model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Sanity‑check generation output:")
    print(response)

if __name__ == "__main__":
    main()
