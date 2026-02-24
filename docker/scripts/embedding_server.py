"""
Embedding microservice for NVIDIA Jetson AGX Thor.

Run: uvicorn embedding_server:app --host 0.0.0.0 --port 8100

Endpoints
---------
POST /embed
    JSON response with embedding as float array.
POST /embed_fast
    Base64 binary response (for production, ~3x faster serialization).
POST /embed_batch
    JSON batch response for multiple queries.
GET /health
    Health check endpoint.
"""

import base64
import logging
import time

from fastapi import FastAPI
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


class QueryRequest(BaseModel):
    """Single query request body."""

    query: str


class QueryResponse(BaseModel):
    """JSON embedding response with float array."""

    embedding: list[float]
    dimension: int
    latency_ms: float


class FastResponse(BaseModel):
    """Base64 binary embedding response for reduced serialization overhead."""

    embedding_b64: str
    dimension: int
    latency_ms: float


class BatchRequest(BaseModel):
    """Batch query request body containing multiple queries."""

    queries: list[str]


class BatchResponse(BaseModel):
    """Batch embedding response with multiple float arrays."""

    embeddings: list[list[float]]
    dimension: int
    count: int
    latency_ms: float


@app.on_event("startup")
def load_model():
    """Load e5-small-v2 model onto GPU and run warmup inference."""
    global model
    logger.info("Loading e5-small-v2 on CUDA...")
    model = SentenceTransformer("intfloat/e5-small-v2", device="cuda")
    for _ in range(5):
        model.encode(["warmup"], normalize_embeddings=True)
    logger.info("Model ready!")


@app.post("/embed", response_model=QueryResponse)
def embed(req: QueryRequest):
    """Embed a single query and return JSON float array."""
    start = time.perf_counter()
    emb = model.encode([f"query: {req.query}"], normalize_embeddings=True).astype(
        "float32"
    )
    latency = (time.perf_counter() - start) * 1000
    logger.info(f'embed | query="{req.query[:50]}" | latency={latency:.1f}ms')
    return QueryResponse(
        embedding=emb[0].tolist(),
        dimension=len(emb[0]),
        latency_ms=round(latency, 2),
    )


@app.post("/embed_fast", response_model=FastResponse)
def embed_fast(req: QueryRequest):
    """Embed a single query and return base64-encoded binary."""
    start = time.perf_counter()
    emb = model.encode([f"query: {req.query}"], normalize_embeddings=True).astype(
        "float32"
    )
    latency = (time.perf_counter() - start) * 1000
    emb_b64 = base64.b64encode(emb[0].tobytes()).decode("ascii")
    logger.info(f'embed_fast | query="{req.query[:50]}" | latency={latency:.1f}ms')
    return FastResponse(
        embedding_b64=emb_b64,
        dimension=len(emb[0]),
        latency_ms=round(latency, 2),
    )


@app.post("/embed_batch", response_model=BatchResponse)
def embed_batch(req: BatchRequest):
    """Embed multiple queries in a single batched request."""
    start = time.perf_counter()
    prefixed = [f"query: {q}" for q in req.queries]
    embs = model.encode(prefixed, normalize_embeddings=True, batch_size=64).astype(
        "float32"
    )
    latency = (time.perf_counter() - start) * 1000
    logger.info(f"embed_batch | count={len(req.queries)} | latency={latency:.1f}ms")
    return BatchResponse(
        embeddings=[e.tolist() for e in embs],
        dimension=len(embs[0]),
        count=len(embs),
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    """Return service health status."""
    return {"status": "ok", "model": "e5-small-v2"}
