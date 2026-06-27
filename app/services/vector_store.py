from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid

# ----------------------------

# Config

# ----------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "rag_chunks"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

MIN_SCORE = 0.30


# ----------------------------
# Collection Management
# ----------------------------


def reset_collection(vector_size: int):
    """
    DEV ONLY: fully reset collection
    """
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print("Created fresh collection")


def create_collection(vector_size: int):
    """
    Create collection if not exists
    """
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ----------------------------
# Ingestion
# ----------------------------


def upsert_chunks(chunks, embeddings):
    """
    Store chunks + embeddings in Qdrant
    """

    points = []

    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ----------------------------
# Vector Search
# ----------------------------


def search(query_vector, top_k: int = 5):
    """
    Dense vector search with:
    - overfetching
    - score filtering
    - deduplication
    """

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
    )

    seen_chunks = set()
    filtered = []

    for r in results.points:
        if r.score < MIN_SCORE:
            continue

        chunk_id = r.payload.get("chunk_id")

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)
        filtered.append(r)

        if len(filtered) == top_k:
            break

    return filtered


# ----------------------------
# Hybrid Fusion (NEW)
# ----------------------------


def hybrid_search(vector_results, bm25_results, top_k: int = 5):
    """
    Fuse BM25 + vector results
    """

    score_map = {}

    # Dense results weight
    for i, r in enumerate(vector_results):
        key = r.payload["chunk_id"]
        score_map[key] = score_map.get(key, 0) + (1 / (i + 1)) * 0.7

    # BM25 results weight
    for i, r in enumerate(bm25_results):
        chunk = r["chunk"]
        key = chunk["chunk_id"]
        score_map[key] = score_map.get(key, 0) + (1 / (i + 1)) * 0.3

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]
