import json, os, random

os.makedirs("data/processed", exist_ok=True)
all_records = []
stats = {}

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def add(instruction, output, source):
    if instruction and output and len(str(output)) > 20:
        all_records.append({
            "instruction": str(instruction).strip(),
            "input": "",
            "output": str(output).strip() + "\n\n*Always consult a qualified doctor.*",
            "source": source,
        })

# Show what files exist
print("Files found in data/raw/:")
for f in os.listdir("data/raw"):
    size = os.path.getsize(f"data/raw/{f}") / 1e6
    print(f"  {f:<50} {size:>8.1f} MB")
print()

# ChatDoctor — supports both .json and .jsonl
for fname in ["chatdoctor.jsonl", "chatdoctor.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            add(r.get("input", r.get("instruction", "")),
                r.get("output", ""), "chatdoctor")
        stats["chatdoctor"] = len(all_records) - before
        print(f"chatdoctor         : {stats['chatdoctor']:>8,}")
        break

# iCliniq
for fname in ["icliniq.jsonl", "icliniq.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            add(r.get("input", ""),
                r.get("answer_icliniq", r.get("output", "")),
                "icliniq")
        stats["icliniq"] = len(all_records) - before
        print(f"icliniq            : {stats['icliniq']:>8,}")
        break

# MedMCQA
for fname in ["medmcqa.jsonl", "medmcqa.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        opt_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
        for r in records:
            q = r.get("question", "")
            correct = r.get(opt_map.get(r.get("cop", 0), "opa"), "")
            exp = r.get("exp", "")
            opts = f"A) {r.get('opa','')}  B) {r.get('opb','')}  C) {r.get('opc','')}  D) {r.get('opd','')}"
            ans = f"Correct answer: {correct}."
            if exp:
                ans += f" {exp}"
            add(f"{q}\n{opts}", ans, "medmcqa")
        stats["medmcqa"] = len(all_records) - before
        print(f"medmcqa            : {stats['medmcqa']:>8,}")
        break

# PubMedQA
for fname in ["pubmedqa.jsonl", "pubmedqa.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            add(r.get("question", ""), r.get("long_answer", ""), "pubmedqa")
        stats["pubmedqa"] = len(all_records) - before
        print(f"pubmedqa           : {stats['pubmedqa']:>8,}")
        break

# MedQA
for fname in ["medqa.jsonl", "medqa.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            q = r.get("question", "")
            opts = r.get("options", {})
            ans_key = r.get("answer_idx", r.get("answer", ""))
            ans_text = opts.get(ans_key, ans_key) if isinstance(opts, dict) else str(ans_key)
            opts_str = "  ".join([f"{k}) {v}" for k,v in opts.items()]) if isinstance(opts, dict) else ""
            add(f"{q}\n{opts_str}", f"Answer: {ans_text}", "medqa")
        stats["medqa"] = len(all_records) - before
        print(f"medqa              : {stats['medqa']:>8,}")
        break

# Mental Health
for fname in ["mental_health.jsonl", "mental_health.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            add(r.get("Context", r.get("input", r.get("question", ""))),
                r.get("Response", r.get("output", r.get("answer", ""))),
                "mental_health")
        stats["mental_health"] = len(all_records) - before
        print(f"mental_health      : {stats['mental_health']:>8,}")
        break

# Medical Reasoning
for fname in ["medical_reasoning.jsonl", "medical_reasoning.json"]:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        records = load_jsonl(path) if fname.endswith(".jsonl") else load_json(path)
        before = len(all_records)
        for r in records:
            add(r.get("instruction", r.get("input", r.get("question", ""))),
                r.get("output", r.get("response", r.get("answer", ""))),
                "medical_reasoning")
        stats["medical_reasoning"] = len(all_records) - before
        print(f"medical_reasoning  : {stats['medical_reasoning']:>8,}")
        break

# Any other jsonl files in raw/
for fname in os.listdir("data/raw"):
    if not fname.endswith(".jsonl"):
        continue
    key = fname.replace(".jsonl", "")
    if key in stats:
        continue
    path = f"data/raw/{fname}"
    records = load_jsonl(path)
    before = len(all_records)
    for r in records:
        q = r.get("instruction", r.get("input", r.get("question", r.get("Patient", ""))))
        a = r.get("output", r.get("answer", r.get("response", r.get("Doctor", ""))))
        add(q, a, key)
    added = len(all_records) - before
    if added > 0:
        stats[key] = added
        print(f"{key:<20} : {added:>8,}")

# Shuffle and split 90/5/5
random.shuffle(all_records)
total = len(all_records)

if total == 0:
    print("\nERROR: No records found! Check data/raw/ has .jsonl files with data.")
else:
    t = int(total * 0.90)
    v = int(total * 0.95)
    train = all_records[:t]
    val   = all_records[t:v]
    test  = all_records[v:]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = f"data/processed/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"""
{'='*45}
TOTAL RECORDS : {total:>10,}
  train.jsonl : {len(train):>10,}
  val.jsonl   : {len(val):>10,}
  test.jsonl  : {len(test):>10,}
{'='*45}
Dataset ready!
""")