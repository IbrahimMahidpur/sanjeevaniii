"""HybridRetriever implementation.

- Dense retrieval using Qdrant (English embeddings).
- Sparse retrieval using a persisted BM25 index.
- Reciprocal Rank Fusion (RRF) to merge results.
- Cross‑encoder reranking.

The class loads the Qdrant client, the BM25 index (pickle file) and the cross‑encoder model on demand.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def _rrf(scores: List[float], k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a single document.
    `scores` is a list of ranks (1‑based) from each source.
    """
    return sum(1 / (rank + k) for rank in scores)


class HybridRetriever:
    def __init__(self,
                 qdrant_url: str = None,
                 collection_name: str = "medical_docs",
                 bm25_path: Path = Path("rag/bm25_index.pkl"),
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection_name
        try:
            import requests
            requests.get(self.qdrant_url, timeout=2)
            self.client = QdrantClient(url=self.qdrant_url)
        except Exception:
            self.client = QdrantClient(path="rag/qdrant_data")
        # Load BM25 index and payload mapping
        if not bm25_path.exists():
            print(f"Warning: BM25 index file not found at {bm25_path}. Sparse retrieval disabled.")
            self.bm25 = None
            self.bm25_payloads = []
        else:
            with bm25_path.open("rb") as f:
                self.bm25, self.bm25_payloads = pickle.load(f)
        # Load dense embedding model (English)
        self.dense_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        # Load cross‑encoder reranker
        self.reranker = SentenceTransformer(cross_encoder_name, device="cpu")

    def _dense_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        try:
            query_vec = self.dense_model.encode([query], normalize_embeddings=True)[0]
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
                with_vector=False,
            )
            # Convert to uniform dict list
            out = []
            for rank, hit in enumerate(results, start=1):
                payload = hit.payload
                payload.update({"_rank": rank, "_source": "dense"})
                out.append(payload)
            return out
        except Exception as e:
            print(f"Warning: Dense search failed: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for rank, idx in enumerate(top_idx, start=1):
            payload = self.bm25_payloads[idx].copy()
            payload.update({"_rank": rank, "_source": "bm25"})
            out.append(payload)
        return out

    def _merge_results(self, dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
        # Index by unique identifier (record_id + chunk_index)
        merged: Dict[str, Dict] = {}
        for src in [dense, sparse]:
            for doc in src:
                uid = f"{doc.get('record_id')}_{doc.get('chunk_index')}"
                rank = doc["_rank"]
                if uid not in merged:
                    merged[uid] = doc.copy()
                    merged[uid]["_ranks"] = [rank]
                else:
                    merged[uid]["_ranks"].append(rank)
        # Compute RRF score
        for uid, doc in merged.items():
            doc["rrf_score"] = _rrf(doc["_ranks"], k=k)
        # Sort by RRF descending
        sorted_docs = sorted(merged.values(), key=lambda d: d["rrf_score"], reverse=True)
        return sorted_docs

    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        # Cross‑encoder expects pairs "query [SEP] document"
        pairs = [f"{query} [SEP] {doc.get('text','')}" for doc in candidates]
        scores = self.reranker.encode(pairs, convert_to_numpy=True, normalize_embeddings=True)
        # Attach scores and sort
        for doc, sc in zip(candidates, scores):
            doc["rerank_score"] = float(sc)
        ranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_k]
        return ranked

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval pipeline.

        Returns a list of dictionaries with at least ``text`` and ``source`` fields.
        """
        dense_hits = self._dense_search(query)
        bm25_hits = self._bm25_search(query)
        merged = self._merge_results(dense_hits, bm25_hits)
        final = self._rerank(query, merged, top_k=top_k)
        # Keep only relevant fields for downstream use
        return [{
            "text": doc.get("text", ""),
            "source": doc.get("source", ""),
            "score": doc.get("rerank_score", 0.0),
        } for doc in final]

# Simple sanity‑check when run as script
if __name__ == "__main__":
    retr = HybridRetriever()
    print(retr.retrieve("diabetes symptoms"))
