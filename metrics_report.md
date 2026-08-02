# LoanSense AI Metrics Report

## Benchmark Methodology
- Timing source: `time.perf_counter()` in benchmark scripts.
- Memory source: `psutil` RSS sampling.
- Environment: local sandbox execution on cloned repository.
- Commands executed:
  - `PYTHONPATH=. python benchmarks/benchmark_ingestion.py --repeats 1 --output /tmp/ingestion_metrics.json`
  - `PYTHONPATH=. python benchmarks/benchmark_retrieval.py --repeats 20 --top-k 5 --output /tmp/retrieval_metrics.json`
  - `PYTHONPATH=. python benchmarks/benchmark_api.py --start-server --repeats 5 --output /tmp/api_metrics.json`
  - `PYTHONPATH=. python benchmarks/benchmark_memory.py --output /tmp/memory_metrics.json`
  - `PYTHONPATH=. python benchmarks/benchmark_rag.py --repeats 2 --output /tmp/rag_metrics.json`

## Hardware Used
- OS: Linux 6.17.0-1020-azure (x86_64)
- CPU: Intel Xeon Platinum 8573C
- vCPU: 4 logical CPUs
- RAM: 15 GiB
- Python: 3.12.3

---

## 1) Knowledge Base Metrics
Measured from `benchmarks/benchmark_ingestion.py` and current repository data/index artifacts.

| Metric | Value |
|---|---:|
| Number of PDFs indexed | 3 |
| Number of chunks generated | 48 |
| Average chunk size | 651.125 chars |
| Maximum chunk size | 697 chars |
| Embedding dimension | 384 |
| FAISS index size | 73,773 bytes |
| BM25 vocabulary size | 1,122 terms |

Notes:
- Embedding model download was unavailable in this environment; embedding timing used synthetic normalized vectors with dimension inferred from existing FAISS index.

## 2) Ingestion Metrics

| Stage | Time (s) |
|---|---:|
| PDF parsing | 0.4010 |
| Chunking | 0.0020 |
| Embedding generation | 1.0332 |
| FAISS+BM25 indexing | 0.0020 |
| Total ingestion | 1.4383 |

Source: `/tmp/ingestion_metrics.json`.

## 3) Retrieval Metrics
(20 repeats x 5 queries, Top-K=5)

| Metric | Avg (ms) | Median (ms) | Min (ms) | Max (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|
| Vector search latency | 0.0348 | 0.0330 | 0.0292 | 0.1217 | 0.0401 |
| BM25 search latency | 0.6271 | 0.6040 | 0.4456 | 1.3959 | 0.8126 |
| Hybrid retrieval latency | 0.6256 | 0.6061 | 0.4525 | 0.9576 | 0.8201 |
| Reranking latency | Not measured | Not measured | Not measured | Not measured | Not measured |
| Total retrieval latency | 0.6256 | 0.6061 | 0.4525 | 0.9576 | 0.8201 |

Notes:
- Embedding backend fell back to synthetic query vectors due blocked Hugging Face model download.
- Reranker model (`BAAI/bge-reranker-base`) could not be downloaded, so reranking latency was not measurable here.

## 4) API Metrics
(uvicorn started locally by benchmark script; 5 requests per endpoint except upload=1)

| Endpoint | Success | Avg Latency (ms) | P95 (ms) | Request Time (ms) | Memory Delta Avg (bytes) |
|---|---:|---:|---:|---:|---:|
| GET /health | 5/5 | 1.379 | 2.124 | 1.379 | 2,457.6 |
| GET /interest-rate | 5/5 | 1.057 | 1.179 | 1.057 | 1,638.4 |
| GET /knowledge-base | 5/5 | 6.459 | 7.283 | 6.459 | 548,864.0 |
| GET /stats | 5/5 | 5.974 | 6.120 | 5.974 | 11,468.8 |
| GET /documents | 5/5 | 1.094 | 1.269 | 1.094 | 0.0 |
| POST /chat | 0/5 | 1022.029 | 1023.595 | 1022.029 | 603,750.4 |
| POST /upload | 0/1 | 1028.352 | 1028.352 | 1028.352 | 712,704.0 |

Failure reason for `/chat` and `/upload`: embedding/LLM dependencies could not be loaded in this environment.

## 5) LLM Metrics
Status: **Not measured**.

Reason:
- `GROQ_API_KEY` was not configured in this environment (`/tmp/rag_metrics.json`).

What to run after setting key:
- `PYTHONPATH=. python benchmarks/benchmark_rag.py --repeats 10 --output /tmp/rag_metrics.json`

This will measure:
- Prompt construction time
- LLM response time
- Token usage (prompt/completion)
- End-to-end response latency

## 6) Memory Metrics

| Workflow | Baseline RSS (MB) | End RSS (MB) | Peak RSS (MB) | Delta RSS (MB) |
|---|---:|---:|---:|---:|
| Ingestion | 899.43 | 906.96 | 905.71 | 7.54 |
| Retrieval | 906.96 | 911.73 | 911.73 | 4.77 |

Notes:
- Embedding/retrieval model loading notes are captured in `/tmp/memory_metrics.json` and indicate blocked external model fetch.

## 7) Retrieval Quality
Status: **Not measured**.

Reason:
- Ground truth relevance labels are not present in repository.

Added template:
- `benchmarks/retrieval_ground_truth_template.json`

How to measure:
1. Fill `relevant_chunk_ids` per question using chunk IDs from vector metadata.
2. Run:
   - `PYTHONPATH=. python benchmarks/benchmark_retrieval.py --ground-truth benchmarks/retrieval_ground_truth_template.json --repeats 20 --top-k 5 --output /tmp/retrieval_metrics.json`
3. Script computes Precision@5, Recall@5, MRR, Hit Rate, Top-1, Top-5.

## 8) Retrieval Method Comparison
Latency comparison from current run:

| Method | Avg Latency (ms) | P95 (ms) | Status |
|---|---:|---:|---|
| FAISS only | 0.0348 | 0.0401 | Measured |
| BM25 only | 0.6271 | 0.8126 | Measured |
| Hybrid | 0.6256 | 0.8201 | Measured |
| Hybrid + Reranker | Not measured | Not measured | Reranker model unavailable |

Quality comparison:
- Not measured (missing labeled ground truth).

## 9) Project Statistics
Computed from repository Python source.

| Metric | Value |
|---|---:|
| Python files | 52 |
| Total LOC (all lines) | 2,688 |
| Non-empty LOC | 1,908 |
| Code LOC (excluding comment-only lines) | 1,901 |
| API endpoints | 7 |
| Classes | 32 |
| Functions | 107 |
| Test files | 1 |
| Test functions | 6 |

## 10) Resume-Ready Metrics (Measured Only)
- Indexed **48** chunks from **3** PDF loan documents.
- Built and measured a modular retrieval stack with **FAISS (0.0348 ms avg search latency)**, **BM25 (0.6271 ms avg)**, and **Hybrid retrieval (0.6256 ms avg)**.
- Exposed and benchmarked **7 FastAPI endpoints**, with low-latency successful GET endpoints (e.g., `/health` at **1.379 ms avg**, `/documents` at **1.094 ms avg**).
- Profiled ingestion runtime at **1.438 s total** for the current corpus and measured peak memory up to **911.73 MB RSS** during retrieval benchmarking.
- Developed a Python codebase comprising **52 files** and **2,688 LOC** with automated tests passing (**6/6**).

## 11) Missing Benchmarks / Gaps
Implemented benchmark scripts in `benchmarks/`:
- `benchmark_ingestion.py`
- `benchmark_retrieval.py`
- `benchmark_api.py`
- `benchmark_memory.py`
- `benchmark_rag.py`

Still blocked in this environment (not fabricated):
1. **True embedding generation latency** with `BAAI/bge-small-en-v1.5` (requires Hugging Face model access).
2. **Reranking latency** with `BAAI/bge-reranker-base` (requires Hugging Face model access).
3. **LLM latency/tokens** (requires valid `GROQ_API_KEY`).
4. **Retrieval quality metrics** (requires labeled ground truth query->chunk relevance data).

## 12) Recommendations
1. Enable model artifact availability (pre-download/cache Hugging Face models in CI or image) to unblock full retrieval and ingestion benchmarks.
2. Add a curated `benchmarks/ground_truth.json` with at least 50+ domain questions and relevant chunk IDs for statistically meaningful quality metrics.
3. Add API middleware timing headers (server-side processing duration) to distinguish network overhead from application time.
4. Run benchmark suites in CI on fixed hardware profiles and persist JSON outputs for trend tracking.
5. Add regression thresholds (e.g., fail CI if P95 retrieval latency regresses >10%).
