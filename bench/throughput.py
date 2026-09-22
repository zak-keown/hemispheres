"""Training-throughput benchmark for MLX on Apple Silicon.

Measures what the experiment plan needs to size its runs:
  1. Peak matmul TFLOPS per dtype (the ceiling for MFU).
  2. Training tokens/sec, peak memory and MFU for GPT models from ~11M to ~700M
     non-embedding parameters, across batch sizes.
  3. Optionally, sustained throughput over minutes to expose thermal throttling.

Each configuration runs in its own subprocess, so an out-of-memory crash only
loses that row and every row starts with a clean allocator.

Usage:
  python bench/throughput.py                  # default sweep, ~10 min
  python bench/throughput.py --quick          # smoke test, ~1 min
  python bench/throughput.py --sizes gpt2,medium --batches 16,32
  python bench/throughput.py --sustain 600    # 10 min on one config, reports per-window tok/s
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
DTYPES = ("float32", "float16", "bfloat16")


# --------------------------------------------------------------------------- workers
# These run inside the subprocess and print a single `RESULT {json}` line.


def worker_matmul(n: int, dtype: str, warmup_s: float = 1.0, window_s: float = 1.0, repeats: int = 5) -> dict:
    import mlx.core as mx

    dt = getattr(mx, dtype)
    a = mx.random.normal((n, n)).astype(dt)
    b = mx.random.normal((n, n)).astype(dt)
    mx.eval(a, b)
    # Run for a while first: short bursts are measured before the GPU clocks settle.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        mx.eval(a @ b)
    rates = []
    for _ in range(repeats):
        iters, t0 = 0, time.perf_counter()
        while (elapsed := time.perf_counter() - t0) < window_s:
            mx.eval(a @ b)
            iters += 1
        rates.append(2 * n**3 * iters / elapsed / 1e12)
    return {"tflops": max(rates), "tflops_median": statistics.median(rates)}


def worker_train(size: str, batch: int, seq: int, dtype: str, vocab: int, compile: bool,
                 warmup: int, steps: int, duration: float, window: float) -> dict:
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from hemispheres.model import GPT, config_for, lm_loss, train_flops_per_token

    cfg = config_for(size, vocab_size=vocab)
    model = GPT(cfg)
    model.set_dtype(getattr(mx, dtype))
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, partial(lm_loss, model))
    state = [model.state, optimizer.state]

    def step(inputs, targets):
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        return loss

    if compile:
        step = mx.compile(step, inputs=state, outputs=state)

    # Random tokens: throughput does not depend on the data.
    tokens = mx.random.randint(0, vocab, (batch, seq + 1))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mx.eval(inputs, targets)

    def timed_step():
        t0 = time.perf_counter()
        loss = step(inputs, targets)
        mx.eval(loss, state)
        return time.perf_counter() - t0, loss.item()

    mx.reset_peak_memory()
    first_step_s, first_loss = timed_step()  # includes graph compilation
    for _ in range(warmup - 1):
        timed_step()

    times, loss = [], first_loss
    t_start = time.perf_counter()
    while (len(times) < steps) if duration <= 0 else (time.perf_counter() - t_start < duration):
        s, loss = timed_step()
        times.append(s)

    tokens_per_step = batch * seq
    median_s = statistics.median(times)
    flops_per_token = train_flops_per_token(cfg, seq)
    out = {
        "params_total": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True),
        "tokens_per_step": tokens_per_step,
        "first_step_s": first_step_s,
        "median_step_s": median_s,
        "tok_per_s": tokens_per_step / median_s,
        "tok_per_s_mean": tokens_per_step * len(times) / sum(times),
        "flops_per_token": flops_per_token,
        "achieved_tflops": flops_per_token * tokens_per_step / median_s / 1e12,
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
        "loss_first": first_loss,
        "loss_last": loss,
        "timed_steps": len(times),
    }
    if duration > 0:
        # Throughput per wall-clock window, to expose thermal throttling.
        windows, acc_t, acc_n = [], 0.0, 0
        for s in times:
            acc_t += s
            acc_n += 1
            if acc_t >= window:
                windows.append(tokens_per_step * acc_n / acc_t)
                acc_t, acc_n = 0.0, 0
        out["window_s"] = window
        out["window_tok_per_s"] = windows
    return out


def worker_main(spec_json: str) -> None:
    spec = json.loads(spec_json)
    kind = spec.pop("kind")
    result = worker_matmul(**spec) if kind == "matmul" else worker_train(**spec)
    print("RESULT " + json.dumps(result), flush=True)


# --------------------------------------------------------------------------- driver


def run_isolated(spec: dict, timeout: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired:
        return {**spec, "error": f"timeout after {timeout:.0f}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return {**spec, **json.loads(line[len("RESULT "):])}
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {**spec, "error": f"exit {proc.returncode}: " + " | ".join(tail)}


def gpu_utilization(samples: int = 5, interval: float = 0.2) -> int | None:
    """Mean GPU 'Device Utilization %' from ioreg, or None if unavailable."""
    import re

    vals = []
    for _ in range(samples):
        try:
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            vals.append(int(m.group(1)))
        time.sleep(interval)
    return round(sum(vals) / len(vals)) if vals else None


def machine_info() -> dict:
    import mlx.core as mx

    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return ""

    gpu_cores = next((line.split(":")[1].strip() for line in sh("system_profiler", "SPDisplaysDataType").splitlines()
                      if "Total Number of Cores" in line), None)
    power = sh("pmset", "-g", "batt").splitlines()
    info = mx.device_info()
    return {
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        "gpu_cores": gpu_cores,
        "memory_gb": info.get("memory_size", 0) / 1e9,
        "gpu_working_set_gb": info.get("max_recommended_working_set_size", 0) / 1e9,
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "power": power[0].strip() if power else None,
        "gpu_util_before_pct": gpu_utilization(),
    }


def build_plan(args) -> tuple[list[dict], list[dict]]:
    matmuls = [{"kind": "matmul", "n": n, "dtype": dt} for dt in DTYPES for n in args.matmul_sizes]
    base = dict(kind="train", seq=args.seq, vocab=args.vocab, warmup=args.warmup, steps=args.steps,
                duration=0.0, window=30.0)
    if args.sustain > 0:
        size = args.sizes[0] if args.sizes_given else "gpt2"
        batch = args.batches[0] if args.batches_given else 16
        train = [{**base, "size": size, "batch": batch, "dtype": "bfloat16", "compile": True,
                  "duration": args.sustain}]
        return [], train
    train = [{**base, "size": s, "batch": b, "dtype": dt, "compile": not args.no_compile}
             for s in args.sizes for b in args.batches for dt in args.dtypes]
    if not (args.quick or args.sizes_given or args.batches_given or args.dtypes_given or args.no_compile):
        # Comparison points that decide how later experiments are run: fp32 vs bf16,
        # compiled vs eager, and the small vocabulary the synthetic-world models will use.
        train += [
            {**base, "size": "gpt2", "batch": 16, "dtype": "float32", "compile": True},
            {**base, "size": "gpt2", "batch": 16, "dtype": "bfloat16", "compile": False},
        ] + [{**base, "size": s, "batch": b, "dtype": "bfloat16", "compile": True, "vocab": 8192}
             for s in ("tiny", "small") for b in (32, 64)]
    return matmuls, train


def fmt_params(n: int) -> str:
    return f"{n / 1e6:.0f}M" if n < 1e9 else f"{n / 1e9:.2f}B"


def summarize(machine: dict, matmuls: list[dict], train: list[dict]) -> str:
    peak = {}
    for r in matmuls:
        if "error" not in r:
            peak[r["dtype"]] = max(peak.get(r["dtype"], 0), r["tflops"])

    lines = [f"# MLX throughput — {machine['chip']} ({machine['gpu_cores']} GPU cores, "
             f"{machine['memory_gb']:.0f} GB)", "",
             f"macOS {machine['macos']} · MLX {machine['mlx']} · {machine['power']} · "
             f"GPU {machine['gpu_util_before_pct']}% busy before start", ""]

    if matmuls:
        lines += ["## Peak matmul", "", "| dtype | n | TFLOPS |", "|---|---|---|"]
        lines += [f"| {r['dtype']} | {r['n']} | " + (f"{r['tflops']:.1f}" if "error" not in r else r["error"]) + " |"
                  for r in matmuls]
        lines.append("")

    lines += ["## Training (AdamW, causal LM, seq %d, random tokens)" % train[0]["seq"], "",
              "| model | vocab | params (non-emb) | batch | dtype | compiled | tok/s | step s | TFLOPS | MFU* | peak mem GB | compile s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in train:
        head = f"| {r['size']} | {r['vocab']} | " + (fmt_params(r["params_total"]) + f" ({fmt_params(r['params_non_embedding'])})"
                                      if "params_total" in r else "—")
        head += f" | {r['batch']} | {r['dtype']} | {'yes' if r['compile'] else 'no'} |"
        if "error" in r:
            lines.append(head + f" {r['error']} | | | | | |")
            continue
        mfu = r["achieved_tflops"] / peak[r["dtype"]] if r["dtype"] in peak else None
        lines.append(head + f" {r['tok_per_s']:,.0f} | {r['median_step_s']:.2f} | {r['achieved_tflops']:.1f} | "
                     + (f"{mfu:.0%}" if mfu else "—") + f" | {r['peak_mem_gb']:.1f} | {r['first_step_s']:.1f} |")
    lines += ["", "\\*MFU is measured against this machine's own peak matmul for the same dtype, not a spec-sheet number.", ""]

    sustained = [r for r in train if "window_tok_per_s" in r]
    for r in sustained:
        w = r["window_tok_per_s"]
        lines += [f"## Sustained: {r['size']} batch {r['batch']} for {r['duration']:.0f}s", "",
                  "Tokens/sec per %.0fs window: " % r["window_s"] + ", ".join(f"{x:,.0f}" for x in w), ""]
        if len(w) >= 2:
            lines += [f"Last window vs first: {w[-1] / w[0]:.0%}", ""]

    best = {}
    for r in train:
        if "error" not in r and r["dtype"] == "bfloat16" and r["compile"]:
            key = f"{r['size']} ({fmt_params(r['params_total'])}, vocab {r['vocab']})"
            if key not in best or r["tok_per_s"] > best[key]["tok_per_s"]:
                best[key] = r
    if best:
        lines += ["## Planning (best bf16 config per size, at peak-burst throughput)", "",
                  "| model | best batch | tok/s | 1B tokens | 20 tok/param (Chinchilla) |", "|---|---|---|---|---|"]
        for key, r in best.items():
            chin = 20 * r["params_total"]
            lines.append(f"| {key} | {r['batch']} | {r['tok_per_s']:,.0f} | "
                         f"{1e9 / r['tok_per_s'] / 3600:.1f} h | {fmt_params(chin)} tok → "
                         f"{chin / r['tok_per_s'] / 3600:.1f} h |")
        lines.append("")
    return "\n".join(lines)


def parse_list(s: str, cast=str) -> list:
    return [cast(x) for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--quick", action="store_true", help="smoke test: tiny+gpt2, one batch size, few steps")
    p.add_argument("--sizes", default=None, help="comma list from hemispheres.model.SIZES (default tiny,small,gpt2,medium)")
    p.add_argument("--batches", default=None, help="comma list of batch sizes (default 8,16,32)")
    p.add_argument("--dtypes", default=None, help="comma list of training dtypes (default bfloat16)")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--matmul-sizes", default="4096,8192")
    p.add_argument("--sustain", type=float, default=0.0, help="seconds to run one config for a throttling test")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="per-config timeout in seconds; configs that overflow memory swap until they hit it")
    p.add_argument("--out", default=None, help="results path stem (default results/throughput-<timestamp>)")
    args = p.parse_args()

    if args.worker:
        worker_main(args.worker)
        return

    args.sizes_given, args.batches_given, args.dtypes_given = (
        args.sizes is not None, args.batches is not None, args.dtypes is not None)
    args.sizes = parse_list(args.sizes or ("tiny,gpt2" if args.quick else "tiny,small,gpt2,medium"))
    args.batches = parse_list(args.batches or ("8" if args.quick else "8,16,32"), int)
    args.dtypes = parse_list(args.dtypes or "bfloat16")
    args.matmul_sizes = parse_list("4096" if args.quick else args.matmul_sizes, int)
    if args.quick:
        args.steps, args.warmup = 5, 2

    machine = machine_info()
    print(f"{machine['chip']} · {machine['gpu_cores']} GPU cores · {machine['memory_gb']:.0f} GB · "
          f"macOS {machine['macos']} · MLX {machine['mlx']}")
    if machine["power"] and "AC Power" not in machine["power"]:
        print("WARNING: not on AC power; results will be throttled.")
    if (machine["gpu_util_before_pct"] or 0) > 10:
        print(f"WARNING: GPU is already {machine['gpu_util_before_pct']}% busy before the benchmark started; "
              "another process is sharing it and results will be unreliable.")

    matmul_plan, train_plan = build_plan(args)
    total = len(matmul_plan) + len(train_plan)
    matmuls, train = [], []
    for i, spec in enumerate(matmul_plan + train_plan, 1):
        label = (f"matmul {spec['dtype']} n={spec['n']}" if spec["kind"] == "matmul" else
                 f"train {spec['size']} b={spec['batch']} {spec['dtype']}{'' if spec['compile'] else ' (eager)'}"
                 + (f" sustain {spec['duration']:.0f}s" if spec["duration"] else ""))
        print(f"[{i}/{total}] {label} ... ", end="", flush=True)
        timeout = args.timeout + spec.get("duration", 0)
        r = run_isolated(spec, timeout)
        if "error" in r:
            print(r["error"])
        elif spec["kind"] == "matmul":
            print(f"{r['tflops']:.1f} TFLOPS")
        else:
            print(f"{r['tok_per_s']:,.0f} tok/s, {r['peak_mem_gb']:.1f} GB peak")
        (matmuls if spec["kind"] == "matmul" else train).append(r)

    report = summarize(machine, matmuls, train)
    print("\n" + report)

    stem = Path(args.out) if args.out else RESULTS_DIR / f"throughput-{datetime.now():%Y%m%d-%H%M}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps({"machine": machine, "matmul": matmuls, "train": train}, indent=2))
    stem.with_suffix(".md").write_text(report)
    print(f"Saved {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
