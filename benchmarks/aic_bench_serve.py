#!/usr/bin/env python3
"""
AIC agentic-style serving benchmark.

Sends ShareGPT multi-turn conversations to a running vLLM OpenAI-compatible
server at a controlled request rate, simulating a realistic agentic workload
with naturally mixed short and long context.

Usage:
    python3 benchmarks/aic_bench_serve.py [options]

Or via make:
    make bench-serve VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct BENCH_RATE=0.5
"""
import argparse, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_DATASET = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_dataset(path: str) -> list:
    if not Path(path).exists():
        print(f"Downloading ShareGPT dataset to {path}...")
        urllib.request.urlretrieve(SHAREGPT_URL, path)
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")
    return data


def conversation_to_messages(conv: dict) -> list[dict] | None:
    """Convert a ShareGPT conversation to OpenAI messages format."""
    messages = []
    for turn in conv.get("conversations", []):
        role = "user" if turn.get("from") in ("human", "user") else "assistant"
        content = turn.get("value", "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    # Must start with user, end with user (so we have something to complete)
    if not messages or messages[0]["role"] != "user":
        return None
    # Keep only up to last user turn (drop trailing assistant turns)
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages if messages else None


def send_request(base_url: str, model: str, messages: list, max_tokens: int,
                 timeout: int) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            latency = time.monotonic() - t0
            usage = body.get("usage", {})
            response_text = ""
            choices = body.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "response_text": response_text,
            }
    except Exception as e:
        return {"ok": False, "latency": time.monotonic() - t0, "error": str(e),
                "response_text": ""}


def main():
    ap = argparse.ArgumentParser(description="AIC agentic serving benchmark")
    ap.add_argument("--base-url", default=os.getenv("AIC_TEST_VLLM_URL", "http://localhost:8000"))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--duration", type=int, default=60, help="Run duration in minutes")
    ap.add_argument("--rate", type=float, default=0.5, help="Target requests/sec")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    ap.add_argument("--concurrency", type=int, default=2, help="Max concurrent requests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="/tmp/aic-bench-results")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Conversation log for Loki/Grafana (tailed by Alloy)
    CONV_LOG = "/tmp/aic-conversations.log"
    conv_log = open(CONV_LOG, "a", buffering=1)

    data = download_dataset(args.dataset)
    # Build prompt pool from conversations with ≥2 turns
    prompts = []
    for conv in data:
        msgs = conversation_to_messages(conv)
        if msgs:
            prompts.append(msgs)
    random.shuffle(prompts)
    print(f"Prompt pool: {len(prompts)} valid conversations")

    end_time = time.monotonic() + args.duration * 60
    interval = 1.0 / args.rate
    results = []
    req_idx = 0

    print(f"\nRunning for {args.duration} min at {args.rate} req/s "
          f"(concurrency={args.concurrency})...")
    print(f"Target: {args.base_url}  Model: {args.model}\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        try:
            while time.monotonic() < end_time:
                t_send = time.monotonic()
                msgs = prompts[req_idx % len(prompts)]
                req_idx += 1
                fut = pool.submit(
                    send_request,
                    args.base_url, args.model, msgs,
                    args.max_tokens, timeout=300,
                )
                futures[fut] = (msgs, time.monotonic())

                # Collect completed futures
                done = [f for f in list(futures) if f.done()]
                for f in done:
                    msgs, r = futures[f][0], f.result()
                    results.append(r)
                    del futures[f]
                    status = "✓" if r["ok"] else "✗"
                    if r["ok"]:
                        print(f"  {status} {len(results):4d}  "
                              f"lat={r['latency']:.1f}s  "
                              f"in={r['prompt_tokens']}  out={r['completion_tokens']}")
                        # Write conversation to Loki-tailed log
                        user_msg = next((m["content"] for m in reversed(msgs)
                                        if m["role"] == "user"), "")
                        conv_log.write(
                            f"[REQ #{len(results)} | {r['prompt_tokens']}in "
                            f"{r['completion_tokens']}out | {r['latency']:.1f}s]\n"
                            f"USER: {user_msg[:500]}\n"
                            f"ASSISTANT: {r['response_text'][:500]}\n"
                            f"---\n"
                        )
                    else:
                        print(f"  {status} {len(results):4d}  ERROR: {r['error'][:60]}")

                # Rate limiting
                elapsed = time.monotonic() - t_send
                sleep = max(0, interval - elapsed)
                time.sleep(sleep)

        except KeyboardInterrupt:
            print("\nInterrupted — waiting for in-flight requests...")

        # Drain remaining
        for f in as_completed(futures, timeout=60):
            r = f.result()
            results.append(r)
    conv_log.close()

    # Summary
    ok = [r for r in results if r["ok"]]
    fail = len(results) - len(ok)
    if ok:
        lats = sorted(r["latency"] for r in ok)
        throughput_in = sum(r["prompt_tokens"] for r in ok)
        throughput_out = sum(r["completion_tokens"] for r in ok)
        elapsed_min = args.duration
        print(f"\n{'='*60}")
        print(f"  Requests:      {len(ok)} OK  {fail} failed")
        print(f"  Throughput:    {len(ok)/elapsed_min/60:.3f} req/s actual")
        print(f"  Prompt tokens: {throughput_in:,}  ({throughput_in/elapsed_min/60:.0f} tok/s)")
        print(f"  Output tokens: {throughput_out:,}  ({throughput_out/elapsed_min/60:.0f} tok/s)")
        print(f"  Latency p50:   {lats[len(lats)//2]:.2f}s")
        print(f"  Latency p95:   {lats[int(len(lats)*0.95)]:.2f}s")
        print(f"  Latency p99:   {lats[int(len(lats)*0.99)]:.2f}s")
        print(f"{'='*60}")

    # Save results
    ts = time.strftime("%Y%m%d-%H%M")
    out_file = f"{args.output}/bench-{ts}.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
