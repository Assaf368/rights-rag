"""Build a Chroma collection per experiment config, incrementally.

Chunk ids are content hashes, so re-running only embeds chunks that are new,
and deletes chunks of the indexed pages that no longer exist. Nothing is
duplicated by a re-run.

    uv run python -m rights_rag.index --strategy structure --headings
"""

import argparse
from collections.abc import Callable, Sequence

import chromadb

from rights_rag.chunking import Chunk, chunk
from rights_rag.corpus import load_corpus

CHROMA_DIR = "chroma"
EMBED_MODEL = "intfloat/multilingual-e5-large"
BATCH = 64

Embedder = Callable[[Sequence[str]], list[list[float]]]


class E5Embedder:
    """multilingual-e5 needs "passage: " / "query: " prefixes; without them
    retrieval quality drops noticeably."""

    def __init__(self, model_name: str = EMBED_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=BATCH,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > BATCH,
        )
        return vectors.tolist()

    def passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" for t in texts])

    def query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]


def collection_name(
    strategy: str, with_headings: bool, model: str = EMBED_MODEL
) -> str:
    model_tag = model.rsplit("/", 1)[-1].removeprefix("multilingual-")
    return f"{strategy}{'_headings' if with_headings else ''}__{model_tag}"


def sync(collection, chunks: list[Chunk], embed: Embedder) -> dict[str, int]:
    """Make the collection hold exactly `chunks` for the pages they come from."""
    doc_ids = sorted({c.metadata["doc_id"] for c in chunks})
    existing = set(
        collection.get(where={"doc_id": {"$in": doc_ids}}, include=[])["ids"]
    )
    wanted = {c.id: c for c in chunks}

    new = [c for cid, c in wanted.items() if cid not in existing]
    stale = sorted(existing - wanted.keys())
    for i in range(0, len(new), BATCH * 8):
        batch = new[i : i + BATCH * 8]
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=embed([c.embed_text for c in batch]),
            documents=[c.text for c in batch],  # clean text, used to verify quotes
            metadatas=[c.metadata | {"embed_text": c.embed_text} for c in batch],
        )
    if stale:
        collection.delete(ids=stale)
    return {"added": len(new), "deleted": len(stale), "kept": len(wanted) - len(new)}


def open_collection(name: str, path: str = CHROMA_DIR):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strategy", choices=["fixed", "unit", "structure"], required=True
    )
    parser.add_argument(
        "--headings", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    chunks = chunk(load_corpus(), args.strategy, with_headings=args.headings)
    name = collection_name(args.strategy, args.headings)
    stats = sync(open_collection(name), chunks, E5Embedder().passages)
    print(f"{name}: {len(chunks)} chunks -> {stats}")


if __name__ == "__main__":
    main()
