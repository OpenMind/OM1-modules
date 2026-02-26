"""
Production QA Engine - uses Docker embedding service for inference.

This module provides a QA retrieval engine that combines a Docker-hosted
embedding service (e5-small-v2) with FAISS vector search to match user
queries against pre-indexed question-answer pairs.
"""

import base64
import logging
import pickle
import time

import faiss
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qa-engine")


class QAEngine:
    """
    QA retrieval engine using FAISS vector search and a remote embedding service.

    Supports two modes:
    - ``query()``: single query via ``/embed`` (base64 binary, lowest latency)
    - ``query_batch()``: batch queries via ``/embed_batch`` (base64, GPU parallel)

    Parameters
    ----------
    embed_base_url : str, optional
        Base URL of the Docker embedding service.
        Default is ``"http://localhost:8100"``.
    index_path : str, optional
        Path to the FAISS index file (``.faiss``).
    data_path : str, optional
        Path to the pickled QA data file (``.pkl``).
    threshold : float, optional
        Minimum cosine similarity score to consider a match.
        Queries below this threshold return ``None`` (fallback to LLM).
        Default is ``0.85``.

    Attributes
    ----------
    index : faiss.IndexFlatIP
        FAISS inner product index for vector search.
    questions : list of str
        All indexed questions.
    answers : list of str
        Corresponding answers (same order as questions).
    threshold : float
        Cosine similarity threshold for matching.
    """

    def __init__(
        self,
        embed_base_url: str = "http://localhost:8100",
        index_path: str = "/home/openmind/Documents/Github/RAG-sys/qa_index_combine.faiss",
        data_path: str = "/home/openmind/Documents/Github/RAG-sys/qa_data_combine.pkl",
        threshold: float = 0.85,
    ):
        logger.info("Loading FAISS index and QA data...")
        self.embed_url = f"{embed_base_url}/embed"
        self.embed_batch_url = f"{embed_base_url}/embed_batch"
        self.index = faiss.read_index(index_path)
        data = pickle.load(open(data_path, "rb"))
        self.questions = data["questions"]
        self.answers = data["answers"]
        self.threshold = threshold

        # Persistent HTTP session (reuses TCP connection)
        self.session = requests.Session()

        logger.info(
            f"Ready. {self.index.ntotal} QA pairs loaded. "
            f"Embedding service: {embed_base_url}"
        )

    def _embed_single(self, query: str) -> tuple[np.ndarray, float]:
        """
        Embed a single query via ``/embed`` (base64 binary).

        Parameters
        ----------
        query : str
            The user query string.

        Returns
        -------
        embedding : np.ndarray
            Shape ``(1, 384)`` float32 embedding vector.
        latency_ms : float
            Round-trip embedding latency in milliseconds.
        """
        start = time.perf_counter()
        resp = self.session.post(self.embed_url, json={"query": query}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        embedding = np.frombuffer(
            base64.b64decode(data["embedding_b64"]), dtype="float32"
        ).reshape(1, -1)
        ms = (time.perf_counter() - start) * 1000
        return embedding, ms

    def _embed_batch(self, queries: list[str]) -> tuple[np.ndarray, float]:
        """
        Embed multiple queries via ``/embed_batch`` (base64 binary).

        Parameters
        ----------
        queries : list of str
            List of user query strings.

        Returns
        -------
        embeddings : np.ndarray
            Shape ``(n, 384)`` float32 embedding matrix.
        latency_ms : float
            Round-trip embedding latency in milliseconds.
        """
        start = time.perf_counter()
        resp = self.session.post(
            self.embed_batch_url, json={"queries": queries}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = np.array(
            [
                np.frombuffer(base64.b64decode(b), dtype="float32")
                for b in data["embeddings_b64"]
            ]
        )
        ms = (time.perf_counter() - start) * 1000
        return embeddings, ms

    def _search(
        self, embedding: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Search FAISS index for nearest neighbors.

        Parameters
        ----------
        embedding : np.ndarray
            Shape ``(n, 384)`` query embedding(s).
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        scores : np.ndarray
            Shape ``(n, top_k)`` cosine similarity scores.
        indices : np.ndarray
            Shape ``(n, top_k)`` indices into the QA data.
        latency_ms : float
            FAISS search latency in milliseconds.
        """
        start = time.perf_counter()
        scores, indices = self.index.search(embedding, top_k)
        ms = (time.perf_counter() - start) * 1000
        return scores, indices, ms

    def query(self, user_query: str, top_k: int = 3) -> dict:
        """
        Process a single query using ``/embed`` for real-time QA.

        Parameters
        ----------
        user_query : str
            The user's natural language question.
        top_k : int, optional
            Number of candidate matches to retrieve. Default is ``3``.

        Returns
        -------
        result : dict
            Dictionary with the following keys:

            - **answer** (*str or None*) -- Best matching answer, or ``None``
              if no match exceeds the threshold (fallback to LLM).
            - **score** (*float*) -- Cosine similarity of the best match.
            - **matched_q** (*str*) -- The matched question from the index.
            - **embed_ms** (*float*) -- Embedding service latency (ms).
            - **faiss_ms** (*float*) -- FAISS search latency (ms).
            - **total_ms** (*float*) -- End-to-end latency (ms).
        """
        total_start = time.perf_counter()

        embedding, embed_ms = self._embed_single(user_query)
        scores, indices, faiss_ms = self._search(embedding, top_k)
        total_ms = (time.perf_counter() - total_start) * 1000

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        matched_q = self.questions[best_idx]
        answer = self.answers[best_idx] if best_score >= self.threshold else None

        status = "HIT" if answer else "MISS"
        logger.info(
            f"{status} | score={best_score:.3f} | embed={embed_ms:.1f}ms | "
            f"faiss={faiss_ms:.3f}ms | total={total_ms:.1f}ms | "
            f'query="{user_query[:50]}"'
        )

        return {
            "answer": answer,
            "score": best_score,
            "matched_q": matched_q,
            "embed_ms": round(embed_ms, 2),
            "faiss_ms": round(faiss_ms, 3),
            "total_ms": round(total_ms, 2),
        }

    def query_batch(self, user_queries: list[str], top_k: int = 3) -> list[dict]:
        """
        Process multiple queries using ``/embed_batch`` for GPU-parallel embedding.

        Parameters
        ----------
        user_queries : list of str
            List of natural language questions.
        top_k : int, optional
            Number of candidate matches per query. Default is ``3``.

        Returns
        -------
        results : list of dict
            List of result dictionaries, one per query. Each dict has the same
            keys as :meth:`query` output. ``embed_ms``, ``faiss_ms``, and
            ``total_ms`` are averaged per query.
        """
        total_start = time.perf_counter()

        embeddings, embed_ms = self._embed_batch(user_queries)
        scores, indices, faiss_ms = self._search(embeddings, top_k)
        total_ms = (time.perf_counter() - total_start) * 1000

        results = []
        for i, q in enumerate(user_queries):
            best_score = float(scores[i][0])
            best_idx = int(indices[i][0])
            matched_q = self.questions[best_idx]
            answer = self.answers[best_idx] if best_score >= self.threshold else None

            status = "HIT" if answer else "MISS"
            logger.info(f'{status} | score={best_score:.3f} | query="{q[:50]}"')

            results.append(
                {
                    "answer": answer,
                    "score": best_score,
                    "matched_q": matched_q,
                    "embed_ms": round(embed_ms / len(user_queries), 2),
                    "faiss_ms": round(faiss_ms / len(user_queries), 3),
                    "total_ms": round(total_ms / len(user_queries), 2),
                }
            )

        logger.info(
            f"BATCH | count={len(user_queries)} | embed={embed_ms:.1f}ms | "
            f"faiss={faiss_ms:.3f}ms | total={total_ms:.1f}ms | "
            f"avg={total_ms/len(user_queries):.1f}ms/query"
        )

        return results


# Quick test
if __name__ == "__main__":
    engine = QAEngine()

    test_queries = [
        "Where is the AWS booth?",
        "Where does the GTC conference will be held?",
        "What sessions are about robotics?",
        "Where is the bathroom?",
        "What is the meaning of life?",
    ]

    # Test single mode
    logger.info("=" * 60)
    logger.info("SINGLE MODE (embed)")
    logger.info("=" * 60)
    for q in test_queries:
        result = engine.query(q)
        status = "HIT" if result["answer"] else "MISS"
        logger.info(
            f"[{status}] Q: {q} | Score: {result['score']:.3f} | "
            f"Total: {result['total_ms']:.1f}ms"
        )
        if result["answer"]:
            logger.info(f"  Answer: {result['answer'][:100]}...")

    # Test batch mode
    logger.info("=" * 60)
    logger.info("BATCH MODE (embed_batch)")
    logger.info("=" * 60)
    results = engine.query_batch(test_queries)
    for q, result in zip(test_queries, results):
        status = "HIT" if result["answer"] else "MISS"
        logger.info(
            f"[{status}] Q: {q} | Score: {result['score']:.3f} | "
            f"Avg: {result['total_ms']:.1f}ms/query"
        )
        if result["answer"]:
            logger.info(f"  Answer: {result['answer'][:100]}...")
