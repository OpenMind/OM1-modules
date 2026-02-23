"""QA Engine - uses Docker embedding service for inference.

Usage:
    from qa_engine import QAEngine
    engine = QAEngine()
    result = engine.query("Where is the AWS booth?")
    print(result["answer"], result["score"], result["total_ms"])
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
    """Fast QA retrieval engine using FAISS and a remote embedding service.

    Loads a pre-built FAISS index and QA data at initialization, then
    for each query: sends the text to the embedding microservice, receives
    a 384-dim vector, and searches the FAISS index for the closest match.

    Parameters
    ----------
    embed_url : str, optional
        URL of the embedding microservice endpoint.
        Default is ``"http://localhost:8100/embed_fast"``.
    index_path : str, optional
        Path to the FAISS index file (``.faiss``).
    data_path : str, optional
        Path to the pickled QA data file (``.pkl``) containing
        ``questions`` and ``answers`` lists.
    threshold : float, optional
        Minimum cosine similarity score to consider a match.
        Below this threshold, ``query()`` returns ``None`` for the answer,
        indicating the query should fall back to an LLM. Default is 0.85.

    Attributes
    ----------
    index : faiss.IndexFlatIP
        FAISS inner-product index for cosine similarity search.
    questions : list[str]
        All indexed question texts.
    answers : list[str]
        Corresponding answer texts (same order as questions).
    session : requests.Session
        Persistent HTTP session for connection reuse.
    """

    def __init__(
        self,
        embed_url: str = "http://localhost:8100/embed_fast",
        index_path: str = "/home/openmind/Documents/Github/RAG-sys/qa_index_combine.faiss",
        data_path: str = "/home/openmind/Documents/Github/RAG-sys/qa_data_combine.pkl",
        threshold: float = 0.85,
    ):
        logger.info("Loading FAISS index and QA data...")
        self.embed_url = embed_url
        self.index = faiss.read_index(index_path)
        data = pickle.load(open(data_path, "rb"))
        self.questions = data["questions"]
        self.answers = data["answers"]
        self.threshold = threshold

        # Keep a persistent HTTP session (reuses TCP connection)
        self.session = requests.Session()

        logger.info(
            f"Ready. {self.index.ntotal} QA pairs loaded. Embedding service: {embed_url}"
        )

    def query(self, user_query: str, top_k: int = 3) -> dict:
        """
        Search for the best matching answer to a user query.

        Sends the query to the embedding service, decodes the base64
        response into a numpy vector, and performs FAISS search.

        Parameters
        ----------
        user_query : str
            The user's natural language question.
        top_k : int, optional
            Number of top matches to retrieve from FAISS. Default is 3.
            Only the best match is used for the answer, but top_k > 1
            can help with debugging.

        Returns
        -------
        dict
            Result dictionary with the following keys:
            answer:     str or None (None = no match, fallback to LLM)
            score:      float (cosine similarity)
            matched_q:  str (the matched question)
            embed_ms:   float (embedding latency)
            faiss_ms:   float (FAISS search latency)
            total_ms:   float (end-to-end latency)
        """
        total_start = time.perf_counter()

        # Get embedding from Docker service (base64 binary)
        embed_start = time.perf_counter()
        resp = self.session.post(self.embed_url, json={"query": user_query}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        embedding = np.frombuffer(
            base64.b64decode(data["embedding_b64"]), dtype="float32"
        ).reshape(1, -1)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # FAISS search
        faiss_start = time.perf_counter()
        scores, indices = self.index.search(embedding, top_k)
        faiss_ms = (time.perf_counter() - faiss_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        matched_q = self.questions[best_idx]

        if best_score >= self.threshold:
            answer = self.answers[best_idx]
            logger.info(
                f"HIT | score={best_score:.3f} | embed={embed_ms:.1f}ms | "
                f"faiss={faiss_ms:.3f}ms | total={total_ms:.1f}ms | "
                f'query="{user_query[:50]}" → "{matched_q[:50]}"'
            )
        else:
            answer = None
            logger.info(
                f"MISS | score={best_score:.3f} | embed={embed_ms:.1f}ms | "
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

    for q in test_queries:
        result = engine.query(q)
        status = "HIT" if result["answer"] else "MISS"
        print(f"\n[{status}] Q: {q}")
        print(f"  Score:     {result['score']:.3f}")
        print(f"  Matched:   {result['matched_q'][:80]}")
        print(f"  Embed:     {result['embed_ms']:.1f}ms")
        print(f"  FAISS:     {result['faiss_ms']:.3f}ms")
        print(f"  Total:     {result['total_ms']:.1f}ms")
        if result["answer"]:
            print(f"  Answer:    {result['answer'][:150]}...")
