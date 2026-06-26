from rank_bm25 import BM25Okapi
from typing import List, Dict
import re


class BM25Retriever:
    def __init__(self):
        self.docs = []
        self.tokenized_docs = []
        self.bm25 = None

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def build(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks
        """
        self.docs = chunks
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5):
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            results.append({"chunk": self.docs[idx], "score": float(score)})

        return results
