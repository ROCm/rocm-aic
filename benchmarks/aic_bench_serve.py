#!/usr/bin/env python3
"""
AIC multi-turn serving benchmark.

Simulates realistic agentic sessions: each session advances a ShareGPT
conversation one turn at a time, waiting a random think-time between turns,
and building the full message history (including actual model responses) so
that LMCache can cache the growing prefix across turns.

Usage:
    python3 benchmarks/aic_bench_serve.py [options]

Or via make:
    make bench-serve VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct BENCH_POOL_SIZE=50
"""
import argparse, json, os, random, sys, time, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_DATASET = os.getenv(
    "AIC_BENCH_DATASET",
    "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json",
)


def download_dataset(path: str) -> list:
    if not os.path.exists(path):
        print(f"Downloading ShareGPT dataset to {path}...")
        urllib.request.urlretrieve(SHAREGPT_URL, path)
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")
    return data


def extract_turns(conv: dict) -> list[dict] | None:
    """Return list of user turns only (content of each human message)."""
    turns = []
    for msg in conv.get("conversations", []):
        if msg.get("from") in ("human", "user"):
            content = msg.get("value", "").strip()
            if content:
                turns.append(content)
    return turns if turns else None


def send_request(base_url: str, model: str, messages: list,
                 max_tokens: int, timeout: int) -> dict:
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
            choices = body.get("choices", [])
            assistant_text = ""
            if choices:
                assistant_text = choices[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "assistant_text": assistant_text,
            }
    except Exception as e:
        return {
            "ok": False,
            "latency": time.monotonic() - t0,
            "error": str(e),
            "assistant_text": "",
        }


class Session:
    """One simulated user session advancing through a multi-turn conversation."""

    def __init__(self, session_id: int, turns: list[str], think_min: float,
                 think_max: float):
        self.session_id = session_id
        self.turns = turns          # user turn texts from the chosen conversation
        self.think_min = think_min
        self.think_max = think_max
        self.history: list[dict] = []   # accumulated messages (user + assistant)
        self.turn_idx = 0
        self.next_send_at = time.monotonic()  # when to send the next turn
        self.total_turns_done = 0

    def ready(self) -> bool:
        return time.monotonic() >= self.next_send_at and self.turn_idx < len(self.turns)

    def next_user_message(self) -> list[dict]:
        """Return the full message list to send (history + next user turn)."""
        user_text = self.turns[self.turn_idx]
        return self.history + [{"role": "user", "content": user_text}]

    def record_response(self, user_text: str, assistant_text: str,
                        latency: float, prompt_tokens: int,
                        completion_tokens: int) -> dict:
        """Append turn to history, schedule next think-time, return result dict."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self.turn_idx += 1
        self.total_turns_done += 1
        think = random.uniform(self.think_min, self.think_max)
        self.next_send_at = time.monotonic() + think
        return {
            "session_id": self.session_id,
            "turn": self.turn_idx,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "think_time": think,
        }

    def reset(self, turns: list[str]) -> None:
        """Restart with a (possibly new) conversation."""
        self.turns = turns
        self.history = []
        self.turn_idx = 0
        self.next_send_at = time.monotonic()


def main():
    ap = argparse.ArgumentParser(description="AIC multi-turn serving benchmark")
    ap.add_argument("--base-url", default=os.getenv("AIC_TEST_VLLM_URL",
                                                      "http://localhost:8000"))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL",
                                                   "Qwen/Qwen2.5-3B-Instruct"))
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--duration", type=int, default=60,
                    help="Run duration in minutes")
    ap.add_argument("--sessions", type=int, default=4,
                    help="Number of concurrent simulated user sessions")
    ap.add_argument("--pool-size", type=int, default=0,
                    help="Limit conversation pool to N (0 = full dataset)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Max output tokens per turn")
    ap.add_argument("--think-min", type=float, default=1.0,
                    help="Min think-time between turns (seconds)")
    ap.add_argument("--think-max", type=float, default=5.0,
                    help="Max think-time between turns (seconds)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (0 = time-based, unique each run)")
    ap.add_argument("--output", default="/tmp/aic-bench-results")
    args = ap.parse_args()

    seed = int(time.time()) if args.seed == 0 else args.seed
    random.seed(seed)
    print(f"Seed: {seed}{' (time-based)' if args.seed == 0 else ''}")
    os.makedirs(args.output, exist_ok=True)

    CONV_LOG = "/tmp/aic-conversations.log"
    conv_log = open(CONV_LOG, "a", buffering=1)

    data = download_dataset(args.dataset)
    convs = [t for c in data if (t := extract_turns(c)) and len(t) >= 2]
    random.shuffle(convs)
    if args.pool_size > 0:
        convs = convs[:args.pool_size]
        print(f"Conversation pool: {len(convs)} (limited by --pool-size)")
    else:
        print(f"Conversation pool: {len(convs)} valid multi-turn conversations")

    # Initialise sessions, each with a random conversation from the pool
    sessions = [
        Session(i, random.choice(convs), args.think_min, args.think_max)
        for i in range(args.sessions)
    ]

    end_time = time.monotonic() + args.duration * 60
    results = []
    lock = threading.Lock()
    total_turns = 0

    print(f"\nRunning {args.sessions} concurrent sessions for {args.duration} min")
    print(f"Think-time: {args.think_min}–{args.think_max}s between turns")
    print(f"Target: {args.base_url}  Model: {args.model}\n")

    with ThreadPoolExecutor(max_workers=args.sessions) as pool:
        futures: dict = {}

        try:
            while time.monotonic() < end_time:
                # Submit ready sessions
                for sess in sessions:
                    if sess in futures:
                        continue  # already in flight
                    if not sess.ready():
                        continue
                    msgs = sess.next_user_message()
                    user_text = sess.turns[sess.turn_idx]
                    fut = pool.submit(
                        send_request,
                        args.base_url, args.model, msgs,
                        args.max_tokens, 300,
                    )
                    futures[fut] = (sess, user_text)

                # Collect completed
                done = [f for f in list(futures) if f.done()]
                for fut in done:
                    sess, user_text = futures.pop(fut)
                    r = fut.result()
                    if r["ok"]:
                        result = sess.record_response(
                            user_text, r["assistant_text"],
                            r["latency"], r["prompt_tokens"],
                            r["completion_tokens"],
                        )
                        with lock:
                            results.append(result)
                            total_turns += 1
                        print(
                            f"  S{sess.session_id:02d} T{sess.turn_idx:02d}"
                            f"  lat={r['latency']:.1f}s"
                            f"  in={r['prompt_tokens']:>5d}"
                            f"  out={r['completion_tokens']:>4d}"
                            f"  think={result['think_time']:.1f}s"
                        )
                        conv_log.write(
                            f"[S{sess.session_id} T{sess.turn_idx} | "
                            f"{r['prompt_tokens']}in {r['completion_tokens']}out | "
                            f"{r['latency']:.1f}s]\n"
                            f"USER: {user_text[:300]}\n"
                            f"ASSISTANT: {r['assistant_text'][:300]}\n---\n"
                        )
                        # If conversation exhausted, restart with a new one
                        if sess.turn_idx >= len(sess.turns):
                            sess.reset(random.choice(convs))
                    else:
                        print(f"  S{sess.session_id:02d} ERROR: {r['error'][:60]}")
                        sess.next_send_at = time.monotonic() + 2.0

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nInterrupted — draining in-flight turns...")

        # Drain in parallel — cancel futures that exceed per-request timeout
        print(f"\nDraining {len(futures)} in-flight turn(s)...")
        for fut in list(futures):
            fut.cancel()  # no-op if already running, prevents queued starts
        from concurrent.futures import as_completed as _ac
        try:
            for fut in _ac(list(futures), timeout=30):
                try:
                    sess, user_text = futures[fut]
                    r = fut.result()
                    if r["ok"]:
                        results.append(sess.record_response(
                            user_text, r["assistant_text"],
                            r["latency"], r["prompt_tokens"], r["completion_tokens"]
                        ))
                except Exception:
                    pass
        except Exception:
            pass

    conv_log.close()

    if results:
        lats = sorted(r["latency"] for r in results)
        total_prompt = sum(r["prompt_tokens"] for r in results)
        total_gen    = sum(r["completion_tokens"] for r in results)
        elapsed_min  = args.duration
        print(f"\n{'='*60}")
        print(f"  Sessions:      {args.sessions}")
        print(f"  Turns:         {len(results)} completed")
        print(f"  Throughput:    {len(results)/elapsed_min/60:.3f} turns/s actual")
        print(f"  Prompt tokens: {total_prompt:,}  ({total_prompt/elapsed_min/60:.0f} tok/s)")
        print(f"  Output tokens: {total_gen:,}  ({total_gen/elapsed_min/60:.0f} tok/s)")
        print(f"  Latency p50:   {lats[len(lats)//2]:.2f}s")
        print(f"  Latency p95:   {lats[int(len(lats)*0.95)]:.2f}s")
        print(f"{'='*60}")

    ts = time.strftime("%Y%m%d-%H%M")
    out_file = f"{args.output}/bench-{ts}.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
