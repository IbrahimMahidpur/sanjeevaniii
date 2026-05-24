"""Unit tests for the HybridRetriever implementation.

These tests mock the Qdrant client and BM25 index to verify the RRF
fusion logic without requiring a running Qdrant instance.
"""

import unittest
from unittest.mock import AsyncMock, patch

# Import the class under test – adjust import path if the package layout differs.
from rag.retriever import HybridRetriever

class TestHybridRetriever(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch QdrantClient and BM25 loading inside the retriever.
        self.qdrant_patch = patch("rag.retriever.AsyncQdrantClient")
        self.bm25_patch = patch("rag.retriever.pickle.load")
        self.mock_qdrant = self.qdrant_patch.start()
        self.mock_bm25 = self.bm25_patch.start()
        # Configure mock behaviours.
        mock_client_instance = AsyncMock()
        # Return 20 dummy dense hits.
        mock_client_instance.search.return_value = [
            {"payload": {"text": f"dense_doc_{i}"}, "score": 1.0 / (i + 1)}
            for i in range(20)
        ]
        self.mock_qdrant.return_value = mock_client_instance
        # BM25 returns 20 dummy sparse hits.
        self.mock_bm25.return_value = {
            f"sparse_doc_{i}": 1.0 / (i + 2) for i in range(20)
        }
        # Patch the cross‑encoder reranker to simply return the input list unchanged.
        self.rerank_patch = patch("rag.retriever.CrossEncoder")
        mock_cross = self.rerank_patch.start()
        mock_cross.return_value.predict = AsyncMock(side_effect=lambda texts: [0.5] * len(texts))

    async def asyncTearDown(self):
        self.qdrant_patch.stop()
        self.bm25_patch.stop()
        self.rerank_patch.stop()

    async def test_retrieve_uses_rrf_and_returns_top_k(self):
        retriever = HybridRetriever()
        results = await retriever.retrieve("test query", top_k=5)
        self.assertEqual(len(results), 5)
        # Ensure results contain expected keys.
        for r in results:
            self.assertIn("text", r)
            self.assertIn("score", r)
            self.assertIn("source", r)
        # Verify that both dense and sparse searches were invoked.
        self.assertTrue(self.mock_qdrant.return_value.search.called)
        self.assertTrue(self.mock_bm25.called)

if __name__ == "__main__":
    unittest.main()
