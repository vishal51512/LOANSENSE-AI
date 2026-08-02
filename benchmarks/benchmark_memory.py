#!/usr/bin/env python3
"""Measure RAM usage during ingestion and retrieval workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("psutil is required. Install with `pip install psutil`.") from exc

from rag.bm25_store import BM25Store
from rag.chunker import ChunkGenerator
from rag.embedding import EmbeddingModel
from rag.hybrid import HybridRetriever
from rag.loader import PDFLoader
from rag.processor import TextProcessor
from rag.reranker import Reranker
from rag.vector_store import VectorStore


PROCESS = psutil.Process()


def rss_bytes() -> int:
    return PROCESS.memory_info().rss


def mb(value: int) -> float:
    return value / (1024 * 1024)


def discover_pdfs(project_root: Path) -> list[Path]:
    files = []
    for folder in (project_root / "data", project_root / "uploads"):
        if folder.is_dir():
            files.extend(sorted(folder.glob("*.pdf")))
    return files


def benchmark_ingestion_memory(project_root: Path) -> dict:
    loader = PDFLoader()
    processor = TextProcessor()
    chunker = ChunkGenerator()
    vector = VectorStore()
    bm25 = BM25Store()

    pdf_paths = discover_pdfs(project_root)
    baseline = rss_bytes()
    peak = baseline

    parsed_pages = []
    for path in pdf_paths:
        parsed_pages.append((path, loader.load_pdf(str(path))))
    peak = max(peak, rss_bytes())

    chunks = []
    for path, pages in parsed_pages:
        cleaned = processor.process(pages)
        chunks.extend(chunker.create_chunks(cleaned, "SBI", "Home Loan", path.name))
    peak = max(peak, rss_bytes())

    embedding_note = None
    try:
        embedder = EmbeddingModel()
        embeddings = embedder.encode_documents([c.text for c in chunks])
        peak = max(peak, rss_bytes())
        vector.build(embeddings, chunks)
        bm25.build(chunks)
        peak = max(peak, rss_bytes())
    except Exception as exc:
        embedding_note = str(exc)

    end = rss_bytes()

    return {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(chunks),
        "baseline_rss_mb": mb(baseline),
        "end_rss_mb": mb(end),
        "peak_rss_mb": mb(peak),
        "delta_rss_mb": mb(end - baseline),
        "embedding_note": embedding_note,
    }


def benchmark_retrieval_memory() -> dict:
    baseline = rss_bytes()
    peak = baseline

    vector = VectorStore()
    vector.load()
    bm25 = BM25Store()
    bm25.load()
    query_execution_note = None
    try:
        embedder = EmbeddingModel()
        hybrid = HybridRetriever(vector, bm25, embedder)
        reranker = Reranker()
        peak = max(peak, rss_bytes())
    except Exception as exc:
        query_execution_note = str(exc)
        hybrid = None
        reranker = None

    queries = [
        "What is the LTV ratio for 20 lacs?",
        "What is SBI floating home loan interest rule?",
        "Does SBI provide fixed rate home loan now?",
    ]

    for query in queries:
        if hybrid is not None and reranker is not None:
            docs = hybrid.search(query, top_k=10)
            _ = reranker.rerank(query, docs, top_k=5)
        else:
            _ = bm25.search(query, top_k=10)
        peak = max(peak, rss_bytes())

    end = rss_bytes()

    return {
        "baseline_rss_mb": mb(baseline),
        "end_rss_mb": mb(end),
        "peak_rss_mb": mb(peak),
        "delta_rss_mb": mb(end - baseline),
        "query_execution_note": query_execution_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAM usage")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "ingestion_memory": benchmark_ingestion_memory(args.project_root),
        "retrieval_memory": benchmark_retrieval_memory(),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
