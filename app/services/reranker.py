from sentence_transformers import CrossEncoder

# lightweight + fast model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, documents: list[dict], top_k: int = 5):
    """
    documents = [{"text": ..., "source": ...}]
    """

    if not documents:
        return []

    pairs = [(query, doc["text"]) for doc in documents]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(documents, scores))

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [{"doc": doc, "score": float(score)} for doc, score in ranked[:top_k]]
