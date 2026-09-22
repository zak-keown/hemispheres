"""Where does a latent-arm training step spend its time?

  python bench/latent_profile.py                 # tiny model, batch 32 × 256, world A's store
  python bench/latent_profile.py --size small --batch 64

Times full compiled training steps (a dense GPT of the same size for reference,
the latent model with plain vs two-stage top-k, and without the retrieval loss),
then the retrieval pieces in isolation at the same shapes: scoring every
position against the store, plain and two-stage top-k, and the key encoder.
"""

import argparse
import statistics
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from hemispheres.data import ExampleSampler, Packer, WorldData, parse_mix
from hemispheres.latent import LatentConfig, LatentGPT, latent_loss, top_k_indices
from hemispheres.model import GPT, SIZES, config_for, masked_lm_loss
from hemispheres.store import Store
from hemispheres.synth.render import Tokenizer


def timed(fn, steps: int, warmup: int = 2) -> float:
    """Median seconds per call; fn must evaluate its own outputs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(steps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def train_step_time(model, loss_fn, batches, steps: int) -> tuple[float, float]:
    opt = optim.AdamW(1e-3)
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(*b):
        out, g = lg(*b)
        opt.update(model, g)
        return out[0] if isinstance(out, tuple) else out

    it = iter(batches * (steps + 3))
    mx.reset_peak_memory()

    def run():
        loss = step(*next(it))
        mx.eval(loss, state)

    return timed(run, steps), mx.get_peak_memory() / 1e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/world-a")
    p.add_argument("--size", default="tiny", choices=sorted(SIZES))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    args = p.parse_args()
    if args.device == "cpu":
        mx.set_default_device(mx.cpu)

    data = WorldData(args.data)
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    vocab = 64 * -(-len(tok) // 64)
    packer = Packer(ExampleSampler(data, "latent", parse_mix(None, "latent")), tok, args.batch, args.seq,
                    store=store, n_reads=3)
    batches = [packer.batch() for _ in range(4)]
    tokens = args.batch * args.seq
    rows = []

    def report(name, s, mem=None):
        rows.append((name, s, mem))
        print(f"{name:<44} {s * 1000:8.1f} ms" + (f"  {tokens / s:10,.0f} tok/s  peak {mem:5.1f} GB" if mem else ""),
              flush=True)

    mx.random.seed(0)
    dense = GPT(config_for(args.size, vocab_size=vocab))
    s, m = train_step_time(dense, partial(masked_lm_loss, dense), [b[:3] for b in batches], args.steps)
    report("train step: dense GPT (reference)", s, m)
    del dense
    mx.clear_cache()

    for name, group, hop_weight in (("train step: latent, two-stage top-k", 256, 0.5),
                                    ("train step: latent, plain top-k", 10**9, 0.5),
                                    ("train step: latent, two-stage, no hop loss", 256, 0.0)):
        mx.random.seed(0)
        model = LatentGPT(LatentConfig(**SIZES[args.size], vocab_size=vocab, retrieval_group=group))
        s, m = train_step_time(model, lambda *b, model=model, w=hop_weight: latent_loss(model, store.arrays, *b,
                                                                                         hop_weight=w),
                               batches, args.steps)
        report(name, s, m)
        mx.clear_cache()

    # Retrieval pieces for one read layer, at the same number of positions.
    cfg = model.cfg
    keys = model.encode_keys(store.arrays)
    q = mx.random.normal((tokens, cfg.key_dim))
    mx.eval(keys, q)
    chunk = cfg.retrieval_chunk

    def scores_only():
        mx.eval([q[c:c + chunk] @ keys.T for c in range(0, tokens, chunk)])

    def topk(group):
        return lambda: mx.eval([top_k_indices(q[c:c + chunk] @ keys.T, cfg.top_k, group)
                                for c in range(0, tokens, chunk)])

    report(f"1 read layer: score {tokens} positions × {len(store)} facts", timed(scores_only, args.steps))
    report("1 read layer: score + plain top-k", timed(topk(10**9), args.steps))
    report("1 read layer: score + two-stage top-k", timed(topk(256), args.steps))

    enc = nn.value_and_grad(model, lambda: model.encode_keys(store.arrays).sum())
    report("key encoder over the whole store (fwd + bwd)", timed(lambda: mx.eval(enc()), args.steps))


if __name__ == "__main__":
    main()
