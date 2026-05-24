"""Build a clean instruction‑tuning dataset.

* Reads all JSONL files under `data/raw/`.
* Deduplicates records by a SHA‑256 hash of the content.
* Formats each record as an instruction‑tuning triple:
    {"instruction": ..., "input": ..., "output": ...}
* Splits into train/val/test with a 90/5/5 ratio.
* Writes the splits to `data/processed/`.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file and return a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line in {path}: {e}", file=sys.stderr)
    return records


def hash_record(rec: Dict) -> str:
    """Create a deterministic hash for deduplication.

    We hash the concatenation of all string values in a sorted key order.
    """
    concat = "".join(str(rec.get(k, "")) for k in sorted(rec.keys()))
    return hashlib.sha256(concat.encode("utf-8")).hexdigest()


def deduplicate(records: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for rec in records:
        h = hash_record(rec)
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    return unique


def format_instruction(rec: Dict) -> Dict:
    """Convert a raw record into the instruction‑tuning schema.

    Expected raw fields (may vary by source):
        - title / abstract (PubMed)
        - text (PDF)
    The function builds a simple instruction based on available keys.
    """
    if rec.get("source") == "pubmed":
        instruction = "Summarize the following medical abstract."
        input_text = rec.get("abstract", "")
        output = rec.get("title", "")
    elif rec.get("source") == "pdf":
        instruction = "Extract the key information from the following medical document."
        input_text = rec.get("text", "")
        output = ""
    else:
        instruction = "Process the following medical text."
        input_text = rec.get("text", rec.get("abstract", ""))
        output = ""
    return {"instruction": instruction, "input": input_text, "output": output}


def split_dataset(records: List[Dict], train_ratio=0.90, val_ratio=0.05) -> Dict[str, List[Dict]]:
    random.shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build instruction‑tuning dataset from raw sources.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write train/val/test splits.",
    )
    args = parser.parse_args()

    # Load all JSONL files
    all_records: List[Dict] = []
    for jsonl_path in args.raw_dir.rglob("*.jsonl"):
        all_records.extend(load_jsonl(jsonl_path))
    print(f"Loaded {len(all_records)} raw records.")

    # Deduplicate
    unique_records = deduplicate(all_records)
    print(f"Deduplicated to {len(unique_records)} records.")

    # Convert to instruction format
    formatted = [format_instruction(r) for r in unique_records]

    # Split
    splits = split_dataset(formatted)
    for split_name, split_records in splits.items():
        out_path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, split_records)
        print(f"Wrote {len(split_records)} records to {out_path}")

    # Simple stats
    total = sum(len(v) for v in splits.values())
    print(f"Dataset creation complete: {total} records total ({len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test).")


if __name__ == "__main__":
    main()
