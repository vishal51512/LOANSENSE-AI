#!/usr/bin/env python3
"""Benchmark document ingestion and report knowledge base metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import faiss

from rag.bm25_store import BM25Store
from rag.chunker import ChunkGenerator
from rag.embedding import EmbeddingModel
from rag.loader import PDFLoader
from rag.processor import TextProcessor
from rag.vector_store import VectorStore


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def discover_pdfs(project_root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for folder_name in ("data", "uploads"):
        folder = project_root / folder_name
        if folder.is_dir():
            pdfs.extend(sorted(folder.glob("*.pdf")))
    return pdfs


def run_once(project_root: Path, pdf_paths: list[Path]) -> dict:
    loader = PDFLoader()
    processor = TextProcessor()
    chunker = ChunkGenerator()
    embedder = EmbeddingModel()

    t_total_start = time.perf_counter()

    t_parse_start = time.perf_counter()
    parsed_pages = []
    for pdf_path in pdf_paths:
        pages = loader.load_pdf(str(pdf_path))
        parsed_pages.append((pdf_path, pages))
    t_parse = time.perf_counter() - t_parse_start

    t_chunk_start = time.perf_counter()
    chunks = []
    for pdf_path, pages in parsed_pages:
        cleaned = processor.process(pages)
        chunks.extend(
            chunker.create_chunks(
                cleaned,
                bank="SBI",
                loan_type="Home Loan",
                document_name=pdf_path.name,
            )
        )
    t_chunk = time.perf_counter() - t_chunk_start

    t_embed_start = time.perf_counter()
    embeddings = embedder.encode_documents([c.text for c in chunks])
    t_embed = time.perf_counter() - t_embed_start

    t_index_start = time.perf_counter()
    vector = VectorStore()
    vector.build(embeddings, chunks)

    bm25 = BM25Store()
    bm25.build(chunks)

    with tempfile.TemporaryDirectory(prefix="loansense-faiss-") as tmp_dir:
        index_path = Path(tmp_dir) / "temp.index"
        faiss.write_index(vector.index, str(index_path))
        faiss_index_size_bytes = index_path.stat().st_size
    t_index = time.perf_counter() - t_index_start

    t_total = time.perf_counter() - t_total_start

    chunk_sizes = [len(c.text) for c in chunks]

    return {
        "timings_seconds": {
            "pdf_parsing": t_parse,
            "chunking": t_chunk,
            "embedding_generation": t_embed,
            "faiss_bm25_indexing": t_index,
            "total_ingestion": t_total,
        },
        "knowledge_base": {
            "pdf_count": len(pdf_paths),
            "chunk_count": len(chunks),
            "average_chunk_size_chars": statistics.fmean(chunk_sizes) if chunk_sizes else 0.0,
            "max_chunk_size_chars": max(chunk_sizes) if chunk_sizes else 0,
            "embedding_dimension": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0,
            "faiss_index_size_bytes": faiss_index_size_bytes,
            "bm25_vocabulary_size": len(getattr(bm25.model, "idf", {})),
        },
    }


def benchmark(project_root: Path, repeats: int) -> dict:
    pdf_paths = discover_pdfs(project_root)
    if not pdf_paths:
        raise FileNotFoundError("No PDFs found under data/ or uploads/.")

    runs = [run_once(project_root, pdf_paths) for _ in range(repeats)]

    timing_keys = list(runs[0]["timings_seconds"].keys())
    timing_summary = {
        key: summarize([run["timings_seconds"][key] for run in runs]) for key in timing_keys
    }

    latest = runs[-1]["knowledge_base"]

    return {
        "project_root": str(project_root),
        "pdfs": [str(p.relative_to(project_root)) for p in pdf_paths],
        "repeats": repeats,
        "knowledge_base": latest,
        "timings_seconds": {
            "runs": [run["timings_seconds"] for run in runs],
            "summary": timing_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ingestion pipeline")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = benchmark(args.project_root, args.repeats)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
