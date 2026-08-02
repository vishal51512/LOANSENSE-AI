#!/usr/bin/env python3
"""Benchmark FastAPI endpoints for latency and memory deltas."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import requests

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


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
        return {"avg": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "avg": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def request_once(method: str, url: str, **kwargs):
    start = time.perf_counter()
    response = requests.request(method=method, url=url, timeout=120, **kwargs)
    elapsed = time.perf_counter() - start
    return response, elapsed


def wait_for_health(base_url: str, timeout_s: int = 60):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok:
                return
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"API did not become healthy within {timeout_s}s. Last error: {last_error}")


def run_endpoint(
    method: str,
    name: str,
    url: str,
    repeats: int,
    process: "psutil.Process | None",
    **kwargs,
) -> dict:
    latencies = []
    rss_samples = []
    errors = []

    for _ in range(repeats):
        before = process.memory_info().rss if process else None
        try:
            resp, elapsed = request_once(method, url, **kwargs)
            latencies.append(elapsed)
            if not resp.ok:
                errors.append({"status": resp.status_code, "body": resp.text[:500]})
        except Exception as exc:
            errors.append({"error": str(exc)})
            continue
        after = process.memory_info().rss if process else None
        if before is not None and after is not None:
            rss_samples.append(after - before)

    out = {
        "endpoint": name,
        "method": method,
        "requests": repeats,
        "successful_requests": len(latencies) - len(errors),
        "latency_seconds": summarize(latencies),
        "request_processing_time_seconds": summarize(latencies),
        "errors": errors,
    }

    if process is None:
        out["memory_usage_bytes"] = {
            "status": "not_measured",
            "reason": "Provide --server-pid to measure API process memory, or use --start-server.",
        }
    else:
        out["memory_usage_bytes"] = summarize(rss_samples)

    return out


def benchmark(base_url: str, repeats: int, server_pid: int | None, start_server: bool) -> dict:
    proc = None
    server_proc = None

    if start_server:
        project_root = Path(__file__).resolve().parents[1]
        server_proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        server_pid = server_proc.pid

    try:
        wait_for_health(base_url)
        if server_pid is not None:
            if psutil is None:
                raise RuntimeError("psutil is required for memory metrics. Install with `pip install psutil`.")
            proc = psutil.Process(server_pid)

        project_root = Path(__file__).resolve().parents[1]
        sample_pdf = project_root / "data" / "home_loan.pdf"

        endpoints = [
            run_endpoint("GET", "/health", f"{base_url}/health", repeats, proc),
            run_endpoint("GET", "/interest-rate", f"{base_url}/interest-rate", repeats, proc),
            run_endpoint("GET", "/knowledge-base", f"{base_url}/knowledge-base", repeats, proc),
            run_endpoint("GET", "/stats", f"{base_url}/stats", repeats, proc),
            run_endpoint("GET", "/documents", f"{base_url}/documents", repeats, proc),
            run_endpoint(
                "POST",
                "/chat",
                f"{base_url}/chat",
                repeats,
                proc,
                json={"question": "What is the current home loan rate?", "bank": "SBI", "loan_type": "Home Loan"},
            ),
            run_endpoint(
                "POST",
                "/upload",
                f"{base_url}/upload",
                1,
                proc,
                files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
            ),
        ]

        return {
            "base_url": base_url,
            "repeats": repeats,
            "server_pid": server_pid,
            "endpoints": endpoints,
        }
    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI endpoints")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = benchmark(
        base_url=args.base_url,
        repeats=args.repeats,
        server_pid=args.server_pid,
        start_server=args.start_server,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
