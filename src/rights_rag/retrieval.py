"""Retrievers over one Chroma collection: vectors, BM25, and both fused with
Reciprocal Rank Fusion (RRF).

BM25 runs on the same `embed_text` the vectors were built from (headings
included), so both retrievers see the same information.
"""

import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

CANDIDATES = 50  # taken from each retriever before fusing
RRF_K = 60  # standard RRF constant: dampens the weight of top ranks

# One-letter Hebrew prefixes (and, the, in, to, from, that, as).
PREFIXES = "והבלמשכ"
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?|[A-Za-z]+|[א-ת]+(?:[\"'׳״][א-ת]+)*")


def tokenize(text: str, strip_prefixes: bool = False) -> list[str]:
    """Words and numbers. With strip_prefixes, a Hebrew word also yields itself
    without up to two leading prefix letters ("והפיצויים" -> "הפיצויים", "פיצויים").
    The original is kept, since these letters are often part of the word."""
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.append(tok)
        if strip_prefixes:
            stem = tok
            for _ in range(2):
                if len(stem) > 3 and stem[0] in PREFIXES:
                    stem = stem[1:]
                    tokens.append(stem)
    return tokens


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists: score(id) = sum of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    """Holds every chunk of a collection in memory for BM25 and for building hits."""

    def __init__(self, collection, query_embedder: Callable[[str], list[float]]):
        data = collection.get(include=["documents", "metadatas"])
        self.collection = collection
        self.embed_query = query_embedder
        self.ids: list[str] = data["ids"]
        self.chunks = {
            cid: {"text": doc, **meta}
            for cid, doc, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        }
        texts = [self.chunks[cid]["embed_text"] for cid in self.ids]
        self._bm25 = {
            strip: BM25Okapi([tokenize(t, strip) for t in texts])
            for strip in (False, True)
        }

    def vector(
        self,
        question: str,
        n: int = CANDIDATES,
        query_vector: list[float] | None = None,
    ) -> list[str]:
        vec = query_vector if query_vector is not None else self.embed_query(question)
        return self.collection.query(query_embeddings=[vec], n_results=n, include=[])[
            "ids"
        ][0]

    def bm25(
        self, question: str, n: int = CANDIDATES, strip_prefixes: bool = False
    ) -> list[str]:
        scores = self._bm25[strip_prefixes].get_scores(
            tokenize(question, strip_prefixes)
        )
        top = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in top if scores[i] > 0]

    def hybrid(
        self,
        question: str,
        strip_prefixes: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[str]:
        return rrf(
            [
                self.vector(question, query_vector=query_vector),
                self.bm25(question, strip_prefixes=strip_prefixes),
            ]
        )

    def hits(self, ids: list[str], n: int) -> list[dict]:
        return [{"chunk_id": cid, **self.chunks[cid]} for cid in ids[:n]]


RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder: reads question and chunk together and scores relevance.
    Far more accurate than comparing two separate vectors, but too slow to run
    on every chunk, so it only reorders the vector search's candidates."""

    def __init__(self, model_name: str = RERANK_MODEL):
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, question: str, candidates: list[tuple[str, str]]) -> list[str]:
        """candidates: (chunk_id, text) pairs. Returns ids, most relevant first."""
        if not candidates:
            return []
        scores = self.model.predict(
            [(question, text) for _, text in candidates], batch_size=16
        )
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i][0] for i in order]
