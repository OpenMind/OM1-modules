"""
Embedding microservice for NVIDIA Jetson AGX Thor.
Run: uvicorn embedding_server:app --host 0.0.0.0 --port 8100.

Endpoints
---------
POST /embed
    Single query embedding (base64 binary response).
POST /embed_batch
    Batch query embedding (base64 binary response).
GET /health
    Health check.
"""

import base64
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embedding-service")

app = FastAPI(title="Embedding Service")
model = None


# Request/Response Models
class QueryRequest(BaseModel):
    """
    Request body for single query embedding.

    Attributes
    ----------
    query : str
        The text to embed.
    """

    query: str


class FastResponse(BaseModel):
    """
    Response body for single query embedding.

    Attributes
    ----------
    embedding_b64 : str
        Base64-encoded float32 byte array (384 × 4 = 1536 bytes).
        Decode with ``np.frombuffer(base64.b64decode(s), dtype="float32")``.
    dimension : int
        Embedding dimension (384 for e5-small-v2).
    latency_ms : float
        Model inference latency in milliseconds (excludes network I/O).
    """

    embedding_b64: str
    dimension: int
    latency_ms: float


class BatchRequest(BaseModel):
    """
    Request body for batch query embedding.

    Attributes
    ----------
    queries : list of str
        List of texts to embed.
    """

    queries: list[str]


class BatchResponse(BaseModel):
    """
    Response body for batch query embedding.

    Attributes
    ----------
    embeddings_b64 : list of str
        List of base64-encoded float32 byte arrays, one per query.
    dimension : int
        Embedding dimension (384 for e5-small-v2).
    count : int
        Number of embeddings returned.
    latency_ms : float
        Model inference latency in milliseconds (excludes network I/O).
    """

    embeddings_b64: list[str]
    dimension: int
    count: int
    latency_ms: float


@app.on_event("startup")
def load_model():
    """
    Load the sentence transformer model onto GPU and run warmup inferences.

    This runs automatically when the FastAPI server starts. The warmup
    ensures CUDA kernels are compiled before the first real request,
    avoiding cold-start latency (~2s → ~6ms).
    """
    global model
    logger.info("Loading e5-small-v2 on CUDA...")
    model = SentenceTransformer("intfloat/e5-small-v2", device="cuda")
    for _ in range(5):
        model.encode(["warmup"], normalize_embeddings=True)
    logger.info("Model ready!")


@app.post("/embed", response_model=FastResponse)
def embed(req: QueryRequest):
    """
    Embed a single query and return base64-encoded vector.

    The query is prefixed with ``"query: "`` before encoding, as
    required by the e5 model family.

    Parameters
    ----------
    req : QueryRequest
        Request body containing the query string.

    Returns
    -------
    FastResponse
        Base64-encoded embedding with dimension and latency info.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service not initialized properly.",
        )

    start = time.perf_counter()
    emb = model.encode([f"query: {req.query}"], normalize_embeddings=True).astype(
        "float32"
    )
    latency = (time.perf_counter() - start) * 1000
    emb_b64 = base64.b64encode(emb[0].tobytes()).decode("ascii")
    logger.info(f'embed | query="{req.query[:50]}" | latency={latency:.1f}ms')
    return FastResponse(
        embedding_b64=emb_b64, dimension=len(emb[0]), latency_ms=round(latency, 2)
    )


@app.post("/embed_batch", response_model=BatchResponse)
def embed_batch(req: BatchRequest):
    """
    Embed multiple queries in a single GPU batch.

    Batch processing is significantly faster than individual requests
    for multiple queries (e.g., 10 queries: ~15ms batch vs ~100ms
    sequential).

    Parameters
    ----------
    req : BatchRequest
        Request body containing a list of query strings.

    Returns
    -------
    BatchResponse
        List of base64-encoded embeddings with count and latency info.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service not initialized properly.",
        )

    start = time.perf_counter()
    prefixed = [f"query: {q}" for q in req.queries]
    embs = model.encode(prefixed, normalize_embeddings=True, batch_size=64).astype(
        "float32"
    )
    latency = (time.perf_counter() - start) * 1000
    embs_b64 = [base64.b64encode(e.tobytes()).decode("ascii") for e in embs]
    logger.info(f"embed_batch | count={len(req.queries)} | latency={latency:.1f}ms")
    return BatchResponse(
        embeddings_b64=embs_b64,
        dimension=len(embs[0]),
        count=len(embs),
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns
    -------
    dict
        ``{"status": "ok", "model": "e5-small-v2"}`` if the service is
        running and the model is loaded.
    """
    return {"status": "ok", "model": "e5-small-v2"}
