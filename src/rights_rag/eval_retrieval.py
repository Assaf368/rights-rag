"""Retrieval evaluation: Hit@k and MRR per collection, checked in code.

A question is a hit at k when every fact of its evidence appears verbatim in
the top k chunks. Raw retrievals are saved, so scores can be recomputed without
loading the model:

    uv run python -m rights_rag.eval_retrieval                       # chunking strategies
    uv run python -m rights_rag.eval_retrieval --experiment hybrid   # vector vs BM25 vs RRF
    uv run python -m rights_rag.eval_retrieval --experiment rerank   # vector vs reranked top-N
    uv run python -m rights_rag.eval_retrieval --rescore eval/runs/<file>.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rights_rag.chunking import FIXED_OVERLAP
from rights_rag.golden import fact_found, load_golden, normalize

KS = (1, 3, 5, 10)
TOP_N = max(KS)
RUNS = Path("eval/runs")
COLLECTIONS = [
    "fixed__e5-large",
    "unit__e5-large",
    "unit_headings__e5-large",
    "structure_headings__e5-large",
]


def _texts_up_to(hits: list[dict], k: int) -> list[str]:
    """Texts of the top k hits, plus the joins of hits that are adjacent fixed-size
    chunks of the same page, so a sentence cut by a chunk boundary still counts."""
    top = hits[:k]
    texts = [normalize(h["text"]) for h in top]
    by_pos = {(h["doc_id"], h["chunk_index"]): h for h in top}
    for h in top:
        nxt = by_pos.get((h["doc_id"], h["chunk_index"] + 1))
        if h["strategy"] == "fixed" and nxt:
            texts.append(normalize(h["text"] + nxt["text"][FIXED_OVERLAP:]))
    return texts


def first_full_rank(question: dict, hits: list[dict]) -> int | None:
    """Smallest k at which all evidence facts are covered, or None."""
    for k in range(1, len(hits) + 1):
        texts = _texts_up_to(hits, k)
        if all(
            any(fact_found(fact, t) for t in texts) for fact in question["evidence"]
        ):
            return k
    return None


def score(questions: list[dict], run: dict[str, dict[str, list[dict]]]) -> dict:
    """run[collection][question_id] -> ranked hits. Unanswerable questions have no
    evidence to find, so they are left out of retrieval metrics."""
    answerable = [q for q in questions if q["evidence"]]
    report = {}
    for collection, per_q in run.items():
        ranks = {q["id"]: first_full_rank(q, per_q[q["id"]]) for q in answerable}
        by_type = defaultdict(list)
        for q in answerable:
            by_type[q["type"]].append(ranks[q["id"]])
        report[collection] = {
            "n": len(answerable),
            **{f"hit@{k}": _hit_rate(ranks.values(), k) for k in KS},
            "mrr": sum(1 / r for r in ranks.values() if r) / len(answerable),
            "by_type_hit@5": {t: _hit_rate(rs, 5) for t, rs in sorted(by_type.items())},
            "ranks": ranks,
        }
    return report


def _hit_rate(ranks, k: int) -> float:
    ranks = list(ranks)
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def retrieve(questions: list[dict], collections: list[str]) -> dict:
    from rights_rag.index import E5Embedder, open_collection

    embedder = E5Embedder()
    vectors = [embedder.query(q["question"]) for q in questions]
    run = {}
    for name in collections:
        col = open_collection(name)
        res = col.query(
            query_embeddings=vectors,
            n_results=TOP_N,
            include=["documents", "metadatas", "distances"],
        )
        run[name] = {
            q["id"]: [
                {
                    "chunk_id": cid,
                    "doc_id": meta["doc_id"],
                    "chunk_index": meta["chunk_index"],
                    "section_path": meta["section_path"],
                    "strategy": meta["strategy"],
                    "distance": dist,
                    "text": text,
                }
                for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True)
            ]
            for q, ids, docs, metas, dists in zip(
                questions,
                res["ids"],
                res["documents"],
                res["metadatas"],
                res["distances"],
                strict=True,
            )
        }
    return run


HYBRID_COLLECTION = "structure_headings__e5-large"
HYBRID_CONFIGS = ["vector", "bm25", "bm25_prefix", "hybrid", "hybrid_prefix"]


def retrieve_hybrid(questions: list[dict], collection: str = HYBRID_COLLECTION) -> dict:
    """Same chunks, different retrievers: one changed thing per row."""
    from rights_rag.index import E5Embedder, open_collection
    from rights_rag.retrieval import Retriever

    embedder = E5Embedder()
    retriever = Retriever(open_collection(collection), embedder.query)
    run: dict[str, dict[str, list[dict]]] = {c: {} for c in HYBRID_CONFIGS}
    for q in questions:
        vec = embedder.query(q["question"])
        ranked = {
            "vector": retriever.vector(q["question"], query_vector=vec),
            "bm25": retriever.bm25(q["question"]),
            "bm25_prefix": retriever.bm25(q["question"], strip_prefixes=True),
            "hybrid": retriever.hybrid(q["question"], query_vector=vec),
            "hybrid_prefix": retriever.hybrid(
                q["question"], strip_prefixes=True, query_vector=vec
            ),
        }
        for config, ids in ranked.items():
            run[config][q["id"]] = retriever.hits(ids, TOP_N)
    return run


RERANK_DEPTHS = (20, 50)


def retrieve_rerank(questions: list[dict], collection: str = HYBRID_COLLECTION) -> dict:
    """Vector candidates reordered by a cross-encoder, at two candidate depths."""
    import time

    from rights_rag.index import E5Embedder, open_collection
    from rights_rag.retrieval import Reranker, Retriever

    embedder = E5Embedder()
    retriever = Retriever(open_collection(collection), embedder.query)
    reranker = Reranker()
    run: dict[str, dict[str, list[dict]]] = {"vector": {}} | {
        f"rerank@{n}": {} for n in RERANK_DEPTHS
    }
    seconds = {n: 0.0 for n in RERANK_DEPTHS}
    for q in questions:
        ids = retriever.vector(q["question"], n=max(RERANK_DEPTHS))
        run["vector"][q["id"]] = retriever.hits(ids, TOP_N)
        for n in RERANK_DEPTHS:
            start = time.perf_counter()
            pairs = [(cid, retriever.chunks[cid]["embed_text"]) for cid in ids[:n]]
            ranked = reranker.rerank(q["question"], pairs)
            seconds[n] += time.perf_counter() - start
            run[f"rerank@{n}"][q["id"]] = retriever.hits(ranked, TOP_N)
    for n in RERANK_DEPTHS:
        print(f"rerank@{n}: {1000 * seconds[n] / len(questions):.0f} ms per question")
    return run


def print_report(report: dict) -> None:
    header = (
        f"{'collection':32}"
        + "".join(f"{'hit@' + str(k):>8}" for k in KS)
        + f"{'mrr':>8}"
    )
    print(header)
    for name, r in report.items():
        print(
            f"{name:32}"
            + "".join(f"{r[f'hit@{k}']:8.2f}" for k in KS)
            + f"{r['mrr']:8.2f}"
        )
    print("\nhit@5 by question type:")
    types = sorted({t for r in report.values() for t in r["by_type_hit@5"]})
    print(f"{'collection':32}" + "".join(f"{t[:10]:>11}" for t in types))
    for name, r in report.items():
        print(
            f"{name:32}"
            + "".join(f"{r['by_type_hit@5'].get(t, 0):11.2f}" for t in types)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split", choices=["dev", "test", "assaf", "all"], default="dev"
    )
    parser.add_argument(
        "--rescore", type=Path, help="score a saved run without retrieving"
    )
    parser.add_argument(
        "--experiment", choices=["chunking", "hybrid", "rerank"], default="chunking"
    )
    args = parser.parse_args()

    saved = None
    if args.rescore:
        saved = json.loads(args.rescore.read_text(encoding="utf-8"))
        args.split = saved["split"]  # score the questions the run was made for
    questions = [
        q for q in load_golden() if args.split == "all" or q["split"] == args.split
    ]
    if saved:
        run = saved["run"]
    else:
        if args.experiment == "hybrid":
            run = retrieve_hybrid(questions)
        elif args.experiment == "rerank":
            run = retrieve_rerank(questions)
        else:
            run = retrieve(questions, COLLECTIONS)
        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        out = RUNS / f"{args.experiment}_{args.split}_{stamp}.json"
        out.write_text(
            json.dumps(
                {"split": args.split, "experiment": args.experiment, "run": run},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"saved raw run to {out}\n")
    print(
        f"split={args.split}, answerable questions={sum(1 for q in questions if q['evidence'])}\n"
    )
    print_report(score(questions, run))


if __name__ == "__main__":
    main()
