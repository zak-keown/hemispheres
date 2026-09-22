# Hemispheres

An LLM that splits **reasoning** from **knowledge**. A knowledge-light reasoner reads from a separate store of facts. You add, change or delete a fact by editing the store, not by retraining the reasoner, and the reasoner uses updated facts in multi-step reasoning as well as a normal LLM uses what it memorized.

Start with [`research/BRIEF.md`](research/BRIEF.md). It covers prior art, the gap, the strongest case against the idea, and the experiment plan. The detailed literature reviews are in [`research/tracks/`](research/tracks/).

## Layout

| Path | What |
|---|---|
| `research/` | Literature review and research brief |
| `hemispheres/model.py` | Dense GPT baseline in MLX (RMSNorm, RoPE, tied embeddings) |
| `bench/throughput.py` | Training-throughput benchmark; sizes every experiment |
| `results/` | Benchmark outputs (JSON + Markdown) |

## Setup

Requires Apple Silicon and Python 3.11+.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

## Throughput benchmark

```sh
.venv/bin/python bench/throughput.py            # full sweep, ~15 min
.venv/bin/python bench/throughput.py --quick    # smoke test, ~1 min
.venv/bin/python bench/throughput.py --sustain 600          # 10-min thermal-throttling test (gpt2, batch 16)
.venv/bin/python bench/throughput.py --sizes medium,large --batches 8,16
```

The benchmark measures peak matmul TFLOPS per dtype. It then trains GPT models from ~11M to ~300M non-embedding parameters (add `large` for ~700M) on random tokens, reporting tokens/sec, MFU against the measured matmul peak, and peak memory. Each configuration runs in its own subprocess, so an out-of-memory crash loses only that row. Run it plugged in.
