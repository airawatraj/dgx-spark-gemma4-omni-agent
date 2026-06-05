#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Gemma 4 Benchmark
Tests TPS, TTFT, concurrent sessions, endpoint health, and usable context.

Usage:
  uv run benchmark/benchmark_speed.py
  uv run benchmark/benchmark_speed.py --skip-context
"""

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime

import requests


COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "-" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)} {unit}")


def make_prompt(n_words):
    """Generate a prompt of approximately n_words words."""
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."


def count_tokens_approx(text):
    """Rough token count: about 1.33 tokens per English word."""
    return int(len(text.split()) * 1.33)


def stream_completion(host, port, model, prompt, max_tokens=200, timeout=120, debug=False):
    """
    Stream a single completion.
    Returns: (ttft_ms, tps, total_tokens, full_text, error)
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    full_text = ""
    usage_tokens = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, None, 0, "", f"HTTP {resp.status_code}: {resp.text[:200]}"

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    break
                if debug:
                    print(f"  RAW: {data[:200]}")

                try:
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage_tokens = chunk["usage"].get("completion_tokens")

                    delta = chunk["choices"][0]["delta"]
                    if debug and not full_text:
                        print(f"  DELTA KEYS: {list(delta.keys())}")
                        print(f"  DELTA: {delta}")

                    text = delta.get("content", "") or ""
                    reasoning = delta.get("reasoning", "") or ""
                    combined = text + reasoning
                    if combined:
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text += combined
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    if debug:
                        print(f"  PARSE ERROR: {exc} on: {data[:100]}")
                    continue

    except requests.exceptions.Timeout:
        return None, None, 0, "", "Timeout"
    except requests.exceptions.ConnectionError:
        return None, None, 0, "", "Connection refused - is vLLM running on the specified port?"
    except Exception as exc:
        return None, None, 0, "", str(exc)

    t_end = time.perf_counter()
    if t_first is None:
        return None, None, 0, full_text, "No tokens generated"

    ttft_ms = (t_first - t_start) * 1000
    generation_time = t_end - t_first
    tokens = usage_tokens if usage_tokens and usage_tokens > 0 else max(1, len(full_text) // 4)
    tps = tokens / generation_time if generation_time > 0 else 0
    return round(ttft_ms), round(tps, 1), tokens, full_text, None


def test_baseline_tps(host, port, model, debug=False):
    header("TEST 1 - Baseline TPS (single session, short prompt)")
    prompt = "Explain quantum entanglement in simple terms."
    runs = 3
    results = []

    print("  Running warmup request (not included in averages)...")
    warmup_ttft, warmup_tps, warmup_tokens, _, warmup_err = stream_completion(
        host, port, model, prompt, max_tokens=300, debug=debug
    )
    if warmup_err:
        print(c(f"  WARN Warmup failed: {warmup_err}", "yellow"))
    else:
        print(
            f"  Warmup: TTFT={c(str(warmup_ttft) + 'ms', 'yellow')}  "
            f"TPS={c(str(warmup_tps), 'green')}  tokens={warmup_tokens}"
        )

    print(f"  Running {runs} consecutive requests...")
    for run in range(runs):
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=300, debug=debug
        )
        if err:
            print(c(f"  FAILED Run {run + 1}: {err}", "red"))
            continue
        results.append((ttft, tps, tokens))
        print(f"  Run {run + 1}: TTFT={c(str(ttft) + 'ms', 'yellow')}  TPS={c(str(tps), 'green')}  tokens={tokens}")
        time.sleep(1)

    if results:
        avg_tps = round(statistics.mean(item[1] for item in results), 1)
        avg_ttft = round(statistics.mean(item[0] for item in results))
        peak_tps = max(item[1] for item in results)
        if warmup_ttft is not None and not warmup_err:
            result_line("Warmup TTFT", warmup_ttft, "ms", "yellow")
        result_line("Average TPS", avg_tps, "tok/s", "green")
        result_line("Peak TPS", peak_tps, "tok/s", "green")
        result_line("Average TTFT (steady state)", avg_ttft, "ms", "yellow")
        return avg_tps, peak_tps

    return 0, 0


def test_tps_vs_length(host, port, model):
    header("TEST 2 - TPS vs Output Length")
    lengths = [50, 150, 300, 600, 1000]
    prompt = "Write a detailed explanation of how transformers work in machine learning."

    print(f"  {'Output tokens'.ljust(18)} {'TPS'.ljust(12)} {'TTFT'}")
    print(f"  {'-' * 44}")

    for max_tok in lengths:
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=max_tok
        )
        if err:
            print(f"  {str(max_tok).ljust(18)} {c('FAILED: ' + err, 'red')}")
        else:
            tps_color = "green" if tps >= 15 else "yellow" if tps >= 10 else "red"
            print(
                f"  {(str(tokens) + ' tok').ljust(18)} "
                f"{c(str(tps) + ' tok/s', tps_color).ljust(20)} {ttft}ms"
            )
        time.sleep(1)


def test_concurrent(host, port, model, max_concurrent=4):
    header("TEST 3 - Concurrent Sessions TPS")
    prompts = [
        "Explain the history of the Roman Empire in detail.",
        "Describe how neural networks learn from data.",
        "What are the key principles of thermodynamics?",
        "Explain the causes and effects of the French Revolution.",
    ]

    for concurrency in range(1, max_concurrent + 1):
        results = [None] * concurrency
        errors = []

        def run_request(idx):
            ttft, tps, tokens, _, err = stream_completion(
                host, port, model, prompts[idx % len(prompts)], max_tokens=200
            )
            if err:
                errors.append(err)
            else:
                results[idx] = (tokens, tps)

        threads = [threading.Thread(target=run_request, args=(i,)) for i in range(concurrency)]
        t_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - t_start

        valid = [result for result in results if result is not None]
        if valid:
            total_tokens = sum(tokens for tokens, _ in valid)
            total_tps = round(total_tokens / elapsed, 1) if elapsed > 0 else 0
            per_session = round(total_tps / concurrency, 1)
            color = "green" if per_session >= 10 else "yellow" if per_session >= 6 else "red"
            print(
                f"  {(str(concurrency) + ' session(s)').ljust(14)} "
                f"total={c(str(total_tps) + ' tok/s', color).ljust(22)} "
                f"per-session={c(str(per_session) + ' tok/s', color)}"
            )
        else:
            print(f"  {(str(concurrency) + ' session(s)').ljust(14)} {c('ALL FAILED: ' + str(errors[0]), 'red')}")

        time.sleep(3)


def test_context_window(host, port, model):
    header("TEST 4 - Context Window Limits")
    print(f"  {c('Testing progressively larger contexts...', 'dim')}")
    print(f"  {'Context tokens'.ljust(20)} {'Result'.ljust(20)} {'TPS'}")
    print(f"  {'-' * 50}")

    sizes = [1024, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 190000]
    last_working = 0

    for size in sizes:
        prompt = make_prompt(int(size * 0.75))
        actual_tokens = count_tokens_approx(prompt)

        ttft, tps, gen_tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=100, timeout=180
        )

        if err:
            lower_error = err.lower()
            if "context" in lower_error or "length" in lower_error or "exceed" in lower_error:
                status = c("FAILED Context exceeded", "red")
            elif "timeout" in lower_error:
                status = c("FAILED Timeout", "red")
            else:
                status = c(f"FAILED {err[:25]}", "red")
            print(f"  ~{(str(actual_tokens) + ' tok'):15} {status}")
            break

        last_working = actual_tokens
        tps_color = "green" if tps >= 12 else "yellow" if tps >= 8 else "red"
        print(f"  ~{(str(actual_tokens) + ' tok'):15} {c('OK', 'green'):20} {c(str(tps) + ' tok/s', tps_color)}")
        time.sleep(2)

    if last_working:
        result_line("Max working context", f"~{last_working:,}", "tokens", "green")
    return last_working


def test_memory_check(host, port):
    header("TEST 5 - vLLM Health & Stats")
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        result_line(
            "Health endpoint",
            "OK" if response.status_code == 200 else f"HTTP {response.status_code}",
            color="green" if response.status_code == 200 else "red",
        )
    except Exception as exc:
        result_line("Health endpoint", f"FAILED: {exc}", color="red")

    try:
        response = requests.get(f"http://{host}:{port}/metrics", timeout=5)
        if response.status_code == 200:
            for line in response.text.split("\n"):
                if "gpu_cache_usage_perc" in line and not line.startswith("#"):
                    pct = round(float(line.split()[-1]) * 100, 1)
                    color = "green" if pct < 80 else "yellow" if pct < 95 else "red"
                    result_line("KV cache used", f"{pct}%", color=color)
                if "num_requests_running" in line and not line.startswith("#"):
                    result_line("Requests running", line.split()[-1])
    except Exception:
        result_line("Metrics endpoint", "not available", color="yellow")


def print_summary(avg_tps, peak_tps, max_context, host, port, model):
    header("SUMMARY")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Endpoint", f"http://{host}:{port}")
    result_line("Model", model)
    print()
    result_line(
        "Average TPS (single session)",
        f"{avg_tps}",
        "tok/s",
        "green" if avg_tps >= 18 else "yellow" if avg_tps >= 14 else "red",
    )
    result_line(
        "Peak TPS (single session)",
        f"{peak_tps}",
        "tok/s",
        "green" if peak_tps >= 20 else "yellow" if peak_tps >= 15 else "red",
    )
    result_line("Max usable context", f"~{max_context:,}" if max_context else "not tested", "tokens")
    print()
    if avg_tps >= 20:
        print(f"  {c('Excellent - Gemma 4 config is working well', 'green')}")
    elif avg_tps >= 15:
        print(f"  {c('Good - solid local throughput', 'yellow')}")
    elif avg_tps >= 10:
        print(f"  {c('Moderate - check for swap or memory pressure', 'yellow')}")
    else:
        print(f"  {c('Below target - investigate endpoint or runtime config', 'red')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DGX Spark Gemma 4 vLLM setup",
        epilog="Run with: uv run benchmark/benchmark_speed.py",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--debug", action="store_true", help="Print raw stream chunks for debugging")
    parser.add_argument("--skip-context", action="store_true", help="Skip context window test")
    parser.add_argument("--skip-concurrent", action="store_true", help="Skip concurrent session test")
    args = parser.parse_args()

    print(f"\n{c('DGX Spark Gemma 4 Benchmark', 'bold')}")
    print(f"{c('Target: ', 'dim')}http://{args.host}:{args.port}  model={args.model}")

    try:
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if response.status_code != 200:
            print(c(f"\nFAILED vLLM not healthy (HTTP {response.status_code}). Is the container running?", "red"))
            sys.exit(1)
    except Exception as exc:
        print(c(f"\nFAILED Cannot reach vLLM: {exc}", "red"))
        print(c("  Make sure spark-brain container is running and port 8000 is open.", "dim"))
        sys.exit(1)

    print(c("  OK vLLM is reachable\n", "green"))

    avg_tps, peak_tps = test_baseline_tps(args.host, args.port, args.model, debug=args.debug)
    test_tps_vs_length(args.host, args.port, args.model)

    if not args.skip_concurrent:
        test_concurrent(args.host, args.port, args.model)

    max_context = 0
    if not args.skip_context:
        max_context = test_context_window(args.host, args.port, args.model)

    test_memory_check(args.host, args.port)
    print_summary(avg_tps, peak_tps, max_context, args.host, args.port, args.model)


if __name__ == "__main__":
    main()
