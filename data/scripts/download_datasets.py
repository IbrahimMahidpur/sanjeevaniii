"""
download_datasets.py — Download all vast medical datasets
Run: python data/scripts/download_datasets.py
"""

import os
import json
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

DATASETS = [
    # (hf_name, config, output_filename, split, max_records)
    ("ruslanmv/ai-medical-chatbot",                    None,          "ai_medical_chatbot",   "train", 250000),
    ("lavita/ChatDoctor-HealthCareMagic-100k",         None,          "chatdoctor",           "train", 100000),
    ("lavita/ChatDoctor-iCliniq",                      None,          "icliniq",              "train", None),
    ("medmcqa",                                        None,          "medmcqa",              "train", None),
    ("pubmed_qa",                                      "pqa_labeled", "pubmedqa",             "train", None),
    ("bigbio/med_qa",                                  "med_qa_en_source", "medqa",           "train", None),
    ("heliosbrahma/mental_health_chatbot_dataset",     None,          "mental_health",        "train", None),
    ("FreedomIntelligence/medical-o1-reasoning-SFT",   None,          "medical_reasoning",    "train", 50000),
]

for hf_name, config, fname, split, max_rec in DATASETS:
    out_path = f"data/raw/{fname}.jsonl"
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size > 1000:
            print(f"  SKIP {fname}.jsonl already exists ({size/1e6:.1f} MB)")
            continue

    print(f"\nDownloading: {hf_name}")
    try:
        ds = load_dataset(hf_name, config, trust_remote_code=True)
        data = ds[split]
        if max_rec:
            data = data.select(range(min(max_rec, len(data))))
        data.to_json(out_path)
        size = os.path.getsize(out_path) / 1e6
        print(f"  Saved {len(data):,} records → {out_path} ({size:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Skipping {fname}, continuing...")

print("\n\nAll done! Files in data/raw/:")
for f in sorted(os.listdir("data/raw")):
    if f.endswith(".jsonl"):
        size = os.path.getsize(f"data/raw/{f}") / 1e6
        print(f"  {f:<45} {size:>8.1f} MB")