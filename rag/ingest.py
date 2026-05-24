"""RAG ingestion script.

Loads processed training data, chunks text, generates GPU embeddings,
stores vectors in Qdrant, and builds a BM25 index.
"""

import argparse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


print("SCRIPT STARTED")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk_record(
    record: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter
) -> List[Dict[str, Any]]:
    """Chunk a single record."""

    instruction = record.get("instruction", "")
    inp = record.get("input", "")
    output = record.get("output", "")
    text_field = record.get("text", "")

    parts = []

    if instruction.strip():
        parts.append(f"Q: {instruction.strip()}")

    if inp.strip():
        parts.append(f"Context: {inp.strip()}")

    if output.strip():
        parts.append(f"A: {output.strip()}")

    if text_field.strip():
        parts.append(text_field.strip())

    source_text = "\n".join(parts).strip()

    if not source_text:
        return []

    chunks = splitter.split_text(source_text)

    out = []

    for i, chunk in enumerate(chunks):

        if not chunk.strip():
            continue

        meta = {
            "record_id": record.get("id", f"{id(record)}_{i}"),
            "chunk_index": i,
            "source": record.get("source", "unknown"),
        }

        out.append({
            "text": chunk,
            "metadata": meta
        })

    return out


def build_bm25_index(docs: List[str]) -> BM25Okapi:
    tokenized = [doc.lower().split() for doc in docs]
    return BM25Okapi(tokenized)


def upsert_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[Dict[str, Any]],
    ids: List[int],
):
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Ingest processed medical data into Qdrant."
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )

    parser.add_argument(
        "--collection",
        default="medical_docs",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------

    train_path = args.processed_dir / "train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}"
        )

    records = []

    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} training records.")

    # -----------------------------------------------------------------------
    # Text Splitter
    # -----------------------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    # -----------------------------------------------------------------------
    # GPU Setup
    # -----------------------------------------------------------------------

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load Embedding Model
    # -----------------------------------------------------------------------

    print("Importing SentenceTransformer...")

    import sentence_transformers

    print(sentence_transformers.__file__)

    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device=device
    )

    # FP16 acceleration
    if device == "cuda":
        model.half()

    print("Model loaded successfully.")

    # -----------------------------------------------------------------------
    # Qdrant
    # -----------------------------------------------------------------------

    try:
        import requests

        requests.get(args.qdrant_url, timeout=2)

        client = QdrantClient(url=args.qdrant_url)

        print(f"Connected to Qdrant at {args.qdrant_url}")

    except Exception:

        print("Qdrant server not running. Using local storage.")

        client = QdrantClient(path="rag/qdrant_data")

    # Recreate collection

    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=qmodels.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=qmodels.Distance.COSINE,
        ),
    )

    # -----------------------------------------------------------------------
    # Chunk Records
    # -----------------------------------------------------------------------

    print("Chunking records...")

    all_chunks = []

    for rec in records:
        chunks = chunk_record(rec, splitter)
        all_chunks.extend(chunks)

    print(f"Generated {len(all_chunks)} chunks.")

    # -----------------------------------------------------------------------
    # Batch Embedding
    # -----------------------------------------------------------------------

    texts = [c["text"] for c in all_chunks]

    print("Generating embeddings on GPU...")

    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    print("Embeddings generated.")

    # -----------------------------------------------------------------------
    # Build Payloads
    # -----------------------------------------------------------------------

    payloads = []
    ids = []
    bm25_docs = []
    bm25_payloads = []

    next_id = 1

    for chunk in all_chunks:

        text = chunk["text"]
        meta = chunk["metadata"]

        payload = {
            "text": text,
            "source": meta["source"],
            "record_id": meta["record_id"],
            "chunk_index": meta["chunk_index"],
            "language": "en",
        }

        payloads.append(payload)

        ids.append(next_id)

        next_id += 1

        bm25_docs.append(text)
        bm25_payloads.append(payload)

    # -----------------------------------------------------------------------
    # Upload to Qdrant
    # -----------------------------------------------------------------------

    print(f"Uploading {len(ids)} vectors to Qdrant...")

    upsert_to_qdrant(
        client=client,
        collection_name=args.collection,
        vectors=vectors,
        payloads=payloads,
        ids=ids,
    )

    print("Upload complete.")

    # -----------------------------------------------------------------------
    # Build BM25
    # -----------------------------------------------------------------------

    print("Building BM25 index...")

    bm25 = build_bm25_index(bm25_docs)

    bm25_path = Path("rag/bm25_index.pkl")

    with bm25_path.open("wb") as f:
        pickle.dump((bm25, bm25_payloads), f)

    print(f"BM25 index saved to {bm25_path}")

    print("INGESTION COMPLETE")


if __name__ == "__main__":
    main()