# Hemispheres

An LLM that splits **reasoning** from **knowledge**. A knowledge-light reasoner reads from a separate store of facts. You add, change or delete a fact by editing the store, not by retraining the reasoner, and the reasoner uses updated facts in multi-step reasoning as well as a normal LLM uses what it memorized.

Start with [`research/BRIEF.md`](research/BRIEF.md). It covers prior art, the gap, the strongest case against the idea, and the experiment plan. The detailed literature reviews are in [`research/tracks/`](research/tracks/).

## Layout

| Path | What |
|---|---|
| `research/` | Literature review and research brief |
| `hemispheres/model.py` | Dense GPT baseline in MLX (RMSNorm, RoPE, tied embeddings) |
| `hemispheres/synth/` | Synthetic-world generator: worlds, question splits, counterfactual edits, renderers |
| `hemispheres/train.py` | Training loop for the dense, lookup and in-context-oracle arms |
| `hemispheres/evaluate.py` | Exact-match evaluation: held-out splits, world swap, counterfactual edits |
| `hemispheres/data.py`, `generate.py`, `checkpoint.py` | Batching, decoding with the store in the loop, run directories |
| `tests/` | Invariant tests for the generator and training plumbing (`pytest`) |
| `bench/throughput.py` | Training-throughput benchmark; sizes every experiment |
| `results/` | Benchmark outputs (JSON + Markdown) |

## Setup

Requires Apple Silicon and Python 3.11+.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q tests
```

## Throughput benchmark

```sh
.venv/bin/python bench/throughput.py            # full sweep, ~15 min
.venv/bin/python bench/throughput.py --quick    # smoke test, ~1 min
.venv/bin/python bench/throughput.py --sustain 600          # 10-min thermal-throttling test (gpt2, batch 16)
.venv/bin/python bench/throughput.py --sizes medium,large --batches 8,16
```

The benchmark measures peak matmul TFLOPS per dtype. It then trains GPT models from ~11M to ~300M non-embedding parameters (add `large` for ~700M) on random tokens, reporting tokens/sec, MFU against the measured matmul peak, and peak memory. Each configuration runs in its own subprocess, so an out-of-memory crash loses only that row. Run it plugged in.

## Synthetic worlds

Step 1 of the plan runs on generated worlds. Each world is a knowledge graph of 10k people, 800 companies, 60 universities, 200 cities and 25 countries, linked by 13 relations such as `works_at`, `mentor`, `hq`, `country` and `capital`. That gives about 62k facts and 32 question paths of 1 to 3 hops.

```sh
.venv/bin/python -m hemispheres.synth.build --name world-a --index 0 --edits 1,100,1000
.venv/bin/python -m hemispheres.synth.build --name world-b --index 1 --edits 1,100,1000
```

- **Worlds with different `--index` values share no entity.** Names are hash-partitioned, so worlds can be generated independently and never collide.
- **All worlds share every token.** Names are built from about 100 shared syllable tokens, and the vocabulary is 544 tokens. A reasoner trained on world A has seen every token world B uses, so a world swap tests unseen *facts*, not unseen embeddings.
- **The same facts render four ways, one per experimental arm:**
  - plain bios and questions (dense);
  - bios and questions with inline `[LOOKUP] subject @relation [RESULT] value [END]` calls, where store-supplied tokens are excluded from the loss (lookup arm);
  - raw triples (latent store);
  - gold facts plus distractors in the prompt (in-context oracle).
- **Splits follow the Grokked Transformers in-distribution vs. out-of-distribution protocol.** 20% of people appear in no multi-hop training question (`test_ood`).
- **Edit sets are nested.** The 100-edit set extends the 1-edit set. Each carries `direct` questions (the edited fact itself), `ripple` questions (multi-hop questions that pass through an edited fact) and `locality` questions (untouched, so their answers must not change). `--edit-relations works_at,mentor,hq` concentrates edits on facts that other questions pass through.

Output goes to `data/<name>/` (git-ignored; regenerate deterministically). Check `samples.txt` first: it shows one example of every rendering, along with which tokens are trained on.

## Training and evaluation

Three arms are implemented. Each trains a reasoner from scratch on world A:

| Arm | Where facts live | Trained on |
|---|---|---|
| `dense` | in the weights | bios + direct QA (answer tokens only) |
| `lookup` | in the store; the model writes `[LOOKUP] subject @relation [RESULT]` and the store answers | bios with lookups + QA with one lookup per hop; store tokens are never trained on |
| `context` | in the prompt (in-context oracle) | QA with gold facts + distractors in the prompt |

```sh
.venv/bin/python -m hemispheres.train --arm lookup --data data/world-a --out runs/lookup-a
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a                      # held-out splits
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-b --sets all           # world swap
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a --edits 100          # edits
```

A dense model can only learn a new world or edits by further training. This is its baseline:

```sh
.venv/bin/python -m hemispheres.train --arm dense --data data/world-a --edits 100 --init runs/dense-a \
    --mix edit_facts=1,edit_qa=1 --steps 200 --eval-sets direct,ripple,locality --out runs/dense-a-k100
```

Scoring is exact match on generated text, never token-by-token scoring. For the lookup arm, `trace` also reports whether every lookup queried the right (subject, relation). Defaults: `small` model (29M), fp32, batch 64 × 256 tokens, 10k steps, AdamW with warmup and cosine decay. Runs checkpoint every 2k steps, and `--resume` continues one.
