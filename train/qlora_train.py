"""QLoRA fine‑tuning entry point.

* Loads hyper‑parameters from ``config.yaml``.
* Instantiates the Mistral‑7B‑Instruct‑v0.3 model with 4‑bit NF4 quantisation.
* Applies LoRA (PEFT) according to the config.
* Uses ``trl``'s ``SFTTrainer`` for supervised fine‑tuning on the instruction‑tuning dataset.
* Supports a ``--dry‑run`` flag that loads the model and runs a single training step
  (useful for verification without GPU load).
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(cfg: Dict[str, Any]):
    # 4‑bit NF4 quantisation configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Apply LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model

def format_chat_examples(example: Dict[str, str]) -> Dict[str, str]:
    """Transform an instruction‑tuning record into the Mistral chat format.

    Expected keys in ``example``: ``instruction``, ``input`` (may be empty), ``output``.
    Returns a dict with ``messages`` suitable for ``trl``'s ``SFTTrainer``.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    # Build a single turn conversation using the Mistral ``[INST]`` token format
    if input_text:
        user_msg = f"[INST] {instruction}\n{input_text} [/INST]"
    else:
        user_msg = f"[INST] {instruction} [/INST]"
    assistant_msg = f" {output}"
    return {"messages": [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]}

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine‑tuning driver for the medical chatbot.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the model and execute a single optimizer step without performing a full epoch.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:", cfg)

    model = build_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets (JSONL) – using the ``load_dataset`` helper with ``json`` format
    data_files = {
        "train": str(Path("data/processed/train.jsonl")),
        "validation": str(Path("data/processed/val.jsonl")),
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # Apply chat formatting
    processed_datasets = raw_datasets.map(format_chat_examples, remove_columns=raw_datasets["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        dataset_text_field="messages",
        max_seq_length=cfg.get("max_seq_length", 2048),
        args=dict(
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            learning_rate=cfg.get("learning_rate", 2e-4),
            num_train_epochs=cfg.get("num_train_epochs", 3),
            logging_steps=cfg.get("logging_steps", 10),
            save_steps=cfg.get("save_steps", 100),
            eval_steps=cfg.get("eval_steps", 100),
            output_dir=cfg.get("output_dir", "./checkpoints"),
            fp16=False,
            bf16=cfg.get("bf16", True),
            report_to=["none"],  # Disable wandb/MLflow by default
        ),
    )

    if args.dry_run:
        # Perform a single forward & backward pass to validate the pipeline
        print("Running dry‑run: one training step …")
        trainer.train(num_train_epochs=0, resume_from_checkpoint=None, max_steps=1)
        print("Dry‑run completed successfully.")
    else:
        trainer.train()
        trainer.save_model(cfg.get("output_dir", "./checkpoints"))
        print(f"Training finished. Model checkpoint saved to {cfg.get('output_dir', './checkpoints')}")

if __name__ == "__main__":
    main()
