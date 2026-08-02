#!/usr/bin/env python3
"""Benchmark retrieval latency and optional retrieval quality metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from rag.bm25_store import BM25Store
from rag.embedding import EmbeddingModel
from rag.hybrid import HybridRetriever
from rag.reranker import Reranker
from rag.vector_store import VectorStore


DEFAULT_QUERIES = [
    "What is the loan to value ratio for loans up to 20 lacs?",
    "Is there a fixed interest rate option?",
    "How does SBI revise floating interest rates?",
    "Can home loan insurance premium be added to the loan amount?",
    "What happens if borrower disagrees with revised interest rate?",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


def run_latency(queries: list[str], repeats: int, top_k: int) -> dict:
    vector = VectorStore()
    vector.load()

    bm25 = BM25Store()
    bm25.load()

    embedder = None
    reranker = None
    embedder_error = None
    reranker_error = None
    try:
        embedder = EmbeddingModel()
        hybrid = HybridRetriever(vector, bm25, embedder)
    except Exception as exc:
        embedder_error = str(exc)
        hybrid = None

    try:
        reranker = Reranker()
    except Exception as exc:
        reranker_error = str(exc)

    vector_latencies = []
    bm25_latencies = []
    hybrid_latencies = []
    rerank_latencies = []
    total_latencies = []

    for _ in range(repeats):
        for query in queries:
            if embedder is not None:
                embedding = embedder.encode_query(query)
            else:
                rng = np.random.default_rng(abs(hash(query)) % (2**32))
                embedding = rng.normal(size=(vector.index.d,)).astype(np.float32)
                embedding /= max(np.linalg.norm(embedding), 1e-12)

            start = time.perf_counter()
            _ = vector.search(embedding, top_k=top_k)
            vector_latencies.append(time.perf_counter() - start)

            start = time.perf_counter()
            _ = bm25.search(query, top_k=top_k)
            bm25_latencies.append(time.perf_counter() - start)

            if hybrid is not None:
                start = time.perf_counter()
                hybrid_docs = hybrid.search(query, top_k=top_k)
                hybrid_elapsed = time.perf_counter() - start
            else:
                start = time.perf_counter()
                vector_results = vector.search(embedding, top_k=top_k)
                bm25_results = bm25.search(query, top_k=top_k)
                merged = [(item["metadata"], item["score"]) for item in vector_results]
                merged.extend((chunk, float(score)) for chunk, score in bm25_results)
                seen = {}
                for chunk, score in merged:
                    seen[chunk.chunk_id] = (chunk, score)
                hybrid_docs = list(seen.values())[:top_k]
                hybrid_elapsed = time.perf_counter() - start
            hybrid_latencies.append(hybrid_elapsed)

            if reranker is not None:
                start = time.perf_counter()
                _ = reranker.rerank(query, hybrid_docs, top_k=top_k)
                rerank_elapsed = time.perf_counter() - start
                rerank_latencies.append(rerank_elapsed)
            else:
                rerank_elapsed = 0.0

            total_latencies.append(hybrid_elapsed + rerank_elapsed)

    return {
        "embedding_backend": "model" if embedder is not None else "synthetic_query_vectors",
        "embedding_backend_note": embedder_error,
        "reranker_backend": "model" if reranker is not None else "not_measured",
        "reranker_backend_note": reranker_error,
        "vector_search_latency_seconds": summarize(vector_latencies),
        "bm25_search_latency_seconds": summarize(bm25_latencies),
        "hybrid_retrieval_latency_seconds": summarize(hybrid_latencies),
        "reranking_latency_seconds": summarize(rerank_latencies),
        "total_retrieval_latency_seconds": summarize(total_latencies),
    }


def load_ground_truth(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_quality(ground_truth: dict, top_k: int) -> dict:
    vector = VectorStore()
    vector.load()
    bm25 = BM25Store()
    bm25.load()
    embedder = EmbeddingModel()
    hybrid = HybridRetriever(vector, bm25, embedder)
    reranker = Reranker()

    methods = {
        "faiss_only": lambda q: [item["metadata"].chunk_id for item in vector.search(embedder.encode_query(q), top_k=top_k)],
        "bm25_only": lambda q: [chunk.chunk_id for chunk, _ in bm25.search(q, top_k=top_k)],
        "hybrid": lambda q: [chunk.chunk_id for chunk, _ in hybrid.search(q, top_k=top_k)],
        "hybrid_plus_reranker": lambda q: [chunk.chunk_id for chunk, _ in reranker.rerank(q, hybrid.search(q, top_k=top_k), top_k=top_k)],
    }

    results = {}

    for method_name, runner in methods.items():
        precisions = []
        recalls = []
        reciprocal_ranks = []
        hit_rates = []
        top1 = []
        top5 = []

        for item in ground_truth.get("queries", []):
            query = item["question"]
            relevant = set(item["relevant_chunk_ids"])
            preds = runner(query)

            hits = [p for p in preds[:top_k] if p in relevant]
            hit_count = len(hits)
            precisions.append(hit_count / top_k if top_k else 0.0)
            recalls.append(hit_count / len(relevant) if relevant else 0.0)
            hit_rates.append(1.0 if hit_count > 0 else 0.0)
            top1.append(1.0 if preds and preds[0] in relevant else 0.0)
            top5.append(1.0 if any(p in relevant for p in preds[:5]) else 0.0)

            rr = 0.0
            for rank, pred in enumerate(preds[:top_k], start=1):
                if pred in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        results[method_name] = {
            "precision_at_5": statistics.fmean(precisions) if precisions else 0.0,
            "recall_at_5": statistics.fmean(recalls) if recalls else 0.0,
            "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "hit_rate": statistics.fmean(hit_rates) if hit_rates else 0.0,
            "top1_accuracy": statistics.fmean(top1) if top1 else 0.0,
            "top5_accuracy": statistics.fmean(top5) if top5 else 0.0,
            "evaluated_queries": len(ground_truth.get("queries", [])),
        }

    return results


def write_ground_truth_template(path: Path) -> None:
    template = {
        "description": "Populate relevant_chunk_ids using chunk IDs from vector_store metadata.",
        "queries": [
            {
                "question": "What is the LTV ratio for loans up to Rs.20 Lacs?",
                "relevant_chunk_ids": ["SBI_home_loan_00001"],
            },
            {
                "question": "Is fixed rate available?",
                "relevant_chunk_ids": ["SBI_home_loan_00003"],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latencies")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--write-template", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.write_template:
        write_ground_truth_template(args.write_template)

    if args.queries_file:
        queries = json.loads(args.queries_file.read_text(encoding="utf-8"))
    else:
        queries = DEFAULT_QUERIES

    report = {
        "queries": queries,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "latency": run_latency(queries=queries, repeats=args.repeats, top_k=args.top_k),
    }

    gt = load_ground_truth(args.ground_truth)
    if gt is None:
        report["quality"] = {
            "status": "not_measured",
            "reason": "Ground truth file not provided. Use --write-template to generate a template and then fill relevant_chunk_ids.",
        }
    else:
        try:
            metrics = evaluate_quality(gt, top_k=args.top_k)
            report["quality"] = {"status": "measured", "metrics": metrics}
        except Exception as exc:
            report["quality"] = {
                "status": "not_measured",
                "reason": (
                    "Quality evaluation requires embedding and reranker models. "
                    f"Current error: {exc}"
                ),
            }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
