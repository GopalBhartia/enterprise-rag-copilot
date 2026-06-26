import time
import uuid
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embeddings import get_embedding
from app.services.vector_store import search
from app.services.bm25 import BM25Retriever
from app.services.llm import generate_answer
from app.services.reranker import rerank
from app.services.trace_store import write_trace

router = APIRouter(prefix="/rag", tags=["RAG"])

TOP_K = 5


# -----------------------------
# Request Schema
# -----------------------------
class AskRequest(BaseModel):
    query: str
    top_k: int = TOP_K


# -----------------------------
# BM25 Setup
# -----------------------------
bm25 = BM25Retriever()
BM25_READY = False


def load_bm25():
    import json
    from pathlib import Path

    global BM25_READY

    path = Path("bm25_chunks.json")

    if not path.exists():
        print("BM25 index not found. Run ingestion first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    bm25.build(chunks)
    BM25_READY = True
    print("BM25 index loaded")


load_bm25()


# -----------------------------
# Helper: normalize vector result
# -----------------------------
def normalize_vector_result(r):
    return {
        "text": r.payload.get("text", ""),
        "source": r.payload.get("source", "unknown"),
        "chunk_id": r.payload.get("chunk_id", ""),
        "score": float(getattr(r, "score", 0.0)),
    }


# -----------------------------
# RAG Endpoint
# -----------------------------
@router.post("/ask")
def rag_ask(request: AskRequest):

    trace_id = str(uuid.uuid4())
    start_time = time.time()

    # -----------------------------
    # 1. Embed query
    # -----------------------------
    query_vector = get_embedding(request.query)

    # -----------------------------
    # 2. Vector retrieval (recall)
    # -----------------------------
    vector_results = search(
        query_vector,
        top_k=request.top_k * 5,
    )

    # -----------------------------
    # 3. BM25 retrieval (recall)
    # -----------------------------
    bm25_results = (
        bm25.search(request.query, top_k=request.top_k * 5) if BM25_READY else []
    )

    # -----------------------------
    # 4. Build candidate pool (HYBRID)
    # -----------------------------
    candidates = []
    seen = set()

    vector_trace = []
    bm25_trace = []

    # vector
    for r in vector_results:
        item = normalize_vector_result(r)
        key = item["chunk_id"] or item["source"]

        if key in seen:
            continue
        seen.add(key)

        candidates.append(item)
        vector_trace.append(item)

    # bm25
    for r in bm25_results:
        chunk = r["chunk"]
        key = chunk.get("chunk_id")

        if key in seen:
            continue
        seen.add(key)

        item = {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", "unknown"),
            "chunk_id": key,
            "score": float(r.get("score", 0.0)),
        }

        candidates.append(item)
        bm25_trace.append(item)

    # -----------------------------
    # 5. RERANK
    # -----------------------------
    reranked = rerank(request.query, candidates, top_k=request.top_k)

    final_docs = []
    reranked_trace = []

    for r in reranked:
        doc = r.get("doc", {})
        final_docs.append(doc)
        reranked_trace.append(
            {
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "score": r.get("score", 0.0),
            }
        )

    # -----------------------------
    # 6. Build LLM context
    # -----------------------------
    contexts = []

    for i, doc in enumerate(final_docs):
        contexts.append(
            {
                "id": i,
                "text": doc.get("text", ""),
                "source": doc.get("source", "unknown"),
                "score": doc.get("score", 0.0),
            }
        )

    # -----------------------------
    # 7. LLM generation
    # -----------------------------
    llm_result = generate_answer(request.query, contexts)

    # -----------------------------
    # 8. citations
    # -----------------------------
    citations = [
        {
            "index": c["id"],
            "source": c["source"],
            "score": c["score"],
        }
        for c in contexts
    ]

    # -----------------------------
    # 9. TRACE STORE (COMPLETE)
    # -----------------------------
    latency_ms = int((time.time() - start_time) * 1000)

    trace_payload = {
        "trace_id": trace_id,
        "query": request.query,
        "retrieval": {
            "vector": vector_trace,
            "bm25": bm25_trace,
        },
        "reranked": reranked_trace,
        "final_context": contexts,
        "answer": llm_result["answer"],
        "latency_ms": latency_ms,
        "top_k": request.top_k,
    }

    write_trace(trace_payload)

    # -----------------------------
    # 10. RESPONSE
    # -----------------------------
    return {
        "trace_id": trace_id,
        "query": request.query,
        "answer": llm_result["answer"],
        "citations": citations,
        "contexts": contexts,
    }
