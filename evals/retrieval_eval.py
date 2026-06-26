from collections import defaultdict

from app.services.embeddings import get_embedding
from app.services.vector_store import search
from app.services.bm25 import BM25Retriever
from app.services.reranker import rerank


# ----------------------------
# Test Set
# ----------------------------
TEST_SET = [
    {
        "query": "what is fastapi",
        "expected": ["fastapi/overview.md"],
    },
    {
        "query": "what is dependency injection in fastapi",
        "expected": ["fastapi/dependencies.md"],
    },
    {
        "query": "how does fastapi handle startup and shutdown",
        "expected": ["fastapi/lifecycle.md"],
    },
    {
        "query": "how does pydantic field validation work in models",
        "expected": ["pydantic/validations.md"],
    },
    {
        "query": "what is docker container",
        "expected": ["docker/overview.md"],
    },
]


TOP_K = 5


# ----------------------------
# BM25 setup
# ----------------------------
bm25 = BM25Retriever()
BM25_READY = False


def load_bm25():
    import json
    from pathlib import Path

    global BM25_READY

    path = Path("bm25_chunks.json")

    if not path.exists():
        print("BM25 index not found.")
        return

    with open(path, "r") as f:
        chunks = json.load(f)

    bm25.build(chunks)
    BM25_READY = True


load_bm25()


# ----------------------------
# Metrics
# ----------------------------
def compute_hit(retrieved, expected):
    return any(r in expected for r in retrieved)


def compute_mrr(retrieved, expected):
    for rank, r in enumerate(retrieved, start=1):
        if r in expected:
            return 1 / rank
    return 0


def recall_at_k(retrieved, expected, k):
    return int(any(r in expected for r in retrieved[:k]))


def noise_rate(retrieved, expected):
    """
    % of retrieved chunks that are irrelevant
    """
    if not retrieved:
        return 0
    relevant = sum(1 for r in retrieved if r in expected)
    return 1 - (relevant / len(retrieved))


# ----------------------------
# RUN EVAL
# ----------------------------
def run_eval():
    print("\n===== RAG EVALUATION START =====\n")

    stats = defaultdict(list)

    for item in TEST_SET:
        query = item["query"]
        expected = item["expected"]

        query_vector = get_embedding(query)

        # ----------------------------
        # 1. VECTOR ONLY
        # ----------------------------
        vector_results = search(query_vector, top_k=TOP_K)
        vector_sources = [r.payload["source"] for r in vector_results]

        # ----------------------------
        # 2. HYBRID (BM25 + VECTOR)
        # ----------------------------
        bm25_results = bm25.search(query, top_k=TOP_K) if BM25_READY else []

        hybrid_pool = list(vector_results)

        for r in bm25_results:
            hybrid_pool.append(r["chunk"])

        hybrid_sources = []
        seen = set()

        for r in hybrid_pool:
            src = r.payload["source"] if hasattr(r, "payload") else r["source"]
            if src in seen:
                continue
            seen.add(src)
            hybrid_sources.append(src)

        # ----------------------------
        # 3. RERANKED (FINAL SYSTEM)
        # ----------------------------
        candidates = []

        for r in hybrid_pool:
            if hasattr(r, "payload"):
                candidates.append(
                    {
                        "text": r.payload.get("text", ""),
                        "source": r.payload.get("source", ""),
                        "chunk_id": r.payload.get("chunk_id", ""),
                    }
                )
            else:
                chunk = r["chunk"]
                candidates.append(
                    {
                        "text": chunk.get("text", ""),
                        "source": chunk.get("source", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                    }
                )

        reranked = rerank(query, candidates, top_k=TOP_K)

        reranked_sources = []
        for r in reranked:
            if isinstance(r, dict) and "doc" in r:
                reranked_sources.append(r["doc"]["source"])
            else:
                reranked_sources.append(r["source"])

        # ----------------------------
        # METRICS
        # ----------------------------
        vector_hit = compute_hit(vector_sources, expected)
        hybrid_hit = compute_hit(hybrid_sources, expected)
        rerank_hit = compute_hit(reranked_sources, expected)

        vector_mrr = compute_mrr(vector_sources, expected)
        hybrid_mrr = compute_mrr(hybrid_sources, expected)
        rerank_mrr = compute_mrr(reranked_sources, expected)

        stats["vector_mrr"].append(vector_mrr)
        stats["hybrid_mrr"].append(hybrid_mrr)
        stats["rerank_mrr"].append(rerank_mrr)

        stats["vector_hit"].append(vector_hit)
        stats["hybrid_hit"].append(hybrid_hit)
        stats["rerank_hit"].append(rerank_hit)

        stats["noise_vector"].append(noise_rate(vector_sources, expected))
        stats["noise_hybrid"].append(noise_rate(hybrid_sources, expected))
        stats["noise_rerank"].append(noise_rate(reranked_sources, expected))

        # ----------------------------
        # PRINT DEBUG
        # ----------------------------
        print("\n-----------------------------")
        print(f"QUERY: {query}")
        print(f"EXPECTED: {expected}")

        print("\nVECTOR:", vector_sources)
        print("HYBRID:", hybrid_sources)
        print("RERANK:", reranked_sources)

        print("\nMRR:")
        print(
            f"vector={vector_mrr:.3f}, hybrid={hybrid_mrr:.3f}, rerank={rerank_mrr:.3f}"
        )

    # ----------------------------
    # FINAL SUMMARY
    # ----------------------------
    print("\n===== FINAL SUMMARY =====")

    def avg(x):
        return sum(x) / len(x)

    print("\nHit Rate:")
    print(f"vector={avg(stats['vector_hit']):.2f}")
    print(f"hybrid={avg(stats['hybrid_hit']):.2f}")
    print(f"rerank={avg(stats['rerank_hit']):.2f}")

    print("\nMRR:")
    print(f"vector={avg(stats['vector_mrr']):.3f}")
    print(f"hybrid={avg(stats['hybrid_mrr']):.3f}")
    print(f"rerank={avg(stats['rerank_mrr']):.3f}")

    print("\nNoise Rate (lower is better):")
    print(f"vector={avg(stats['noise_vector']):.2f}")
    print(f"hybrid={avg(stats['noise_hybrid']):.2f}")
    print(f"rerank={avg(stats['noise_rerank']):.2f}")


if __name__ == "__main__":
    run_eval()
