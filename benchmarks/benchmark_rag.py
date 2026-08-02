#!/usr/bin/env python3
"""Benchmark end-to-end RAG latency and LLM metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from llm.prompt_builder import PromptBuilder
from rag.rag_engine import RAGEngine


DEFAULT_QUERIES = [
    "What is the loan to value ratio up to 20 lacs?",
    "What are the rules for floating interest changes?",
    "Can fixed interest rate be selected?",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG + LLM timings")
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if args.queries_file:
        queries = json.loads(args.queries_file.read_text(encoding="utf-8"))
    else:
        queries = DEFAULT_QUERIES

    if not api_key:
        report = {
            "status": "not_measured",
            "reason": "GROQ_API_KEY is not set. Set it in .env to measure LLM response time and token usage.",
            "queries": queries,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    try:
        rag = RAGEngine()
    except Exception as exc:
        report = {
            "status": "not_measured",
            "reason": (
                "RAG model dependencies could not be loaded. "
                "Ensure Hugging Face model download access is available. "
                f"Current error: {exc}"
            ),
            "queries": queries,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    groq_client = Groq(api_key=api_key)
    model = "llama-3.3-70b-versatile"

    prompt_times = []
    llm_times = []
    e2e_times = []
    input_tokens = []
    output_tokens = []

    for _ in range(args.repeats):
        for query in queries:
            e2e_start = time.perf_counter()
            docs = rag.search(query, top_k=5)

            prompt_start = time.perf_counter()
            system_prompt, prompt = PromptBuilder.build(query, docs)
            prompt_times.append(time.perf_counter() - prompt_start)

            llm_start = time.perf_counter()
            response = groq_client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            llm_times.append(time.perf_counter() - llm_start)

            usage = getattr(response, "usage", None)
            if usage is not None:
                if usage.prompt_tokens is not None:
                    input_tokens.append(int(usage.prompt_tokens))
                if usage.completion_tokens is not None:
                    output_tokens.append(int(usage.completion_tokens))

            e2e_times.append(time.perf_counter() - e2e_start)

    report = {
        "status": "measured",
        "queries": queries,
        "repeats": args.repeats,
        "model": model,
        "prompt_construction_time_seconds": summarize(prompt_times),
        "llm_response_time_seconds": summarize(llm_times),
        "end_to_end_response_latency_seconds": summarize(e2e_times),
        "tokens": {
            "prompt_tokens_avg": statistics.fmean(input_tokens) if input_tokens else None,
            "completion_tokens_avg": statistics.fmean(output_tokens) if output_tokens else None,
            "samples": len(output_tokens),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
