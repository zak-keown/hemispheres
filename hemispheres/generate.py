"""Batched greedy decoding, with an optional knowledge store in the loop.

When a store is given and the model emits `[RESULT]`, the query written since
the last `[LOOKUP]` (subject tokens followed by an `@relation` token) is sent to
the store, and the store's answer plus `[END]` is appended to that sequence.
The model never produces store tokens itself.

There is no KV cache: every step re-runs the full forward pass. Sequences are
right-padded, which is safe for a causal model because padding only follows
the positions that are read.
"""

import mlx.core as mx

from .synth.render import Tokenizer
from .synth.schema import END, EOS, LOOKUP, PAD, RESULT
from .synth.world import World


def store_reply(tokens: list[str], store: World) -> list[str]:
    """The store's answer to the lookup query that ends `tokens` (which ends in [RESULT])."""
    try:
        start = len(tokens) - 1 - tokens[::-1].index(LOOKUP)
    except ValueError:
        return [END]
    query = tokens[start + 1:-1]
    if len(query) < 2 or not query[-1].startswith("@"):
        return [END]
    value = store.lookup(tuple(query[:-1]), query[-1][1:])
    return [*value, END] if value else [END]


def generate(model, tok: Tokenizer, prompts: list[list[str]], max_new: int = 64, store: World | None = None,
             batch_size: int = 256) -> list[list[str]]:
    """Greedy continuations of `prompts` (token lists); each stops at [EOS] or `max_new` tokens."""
    out = []
    for i in range(0, len(prompts), batch_size):
        out += _generate_batch(model, tok, prompts[i:i + batch_size], max_new, store)
    return out


def _generate_batch(model, tok: Tokenizer, prompts, max_new, store):
    pad, eos, result = tok.token_id(PAD), tok.token_id(EOS), tok.token_id(RESULT)
    seqs = [tok.encode(p) for p in prompts]
    starts = [len(s) for s in seqs]
    done = [False] * len(seqs)
    n_vocab = len(tok)
    while True:
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        width = max(len(seqs[i]) for i in active)
        batch = mx.array([seqs[i] + [pad] * (width - len(seqs[i])) for i in active], dtype=mx.int32)
        logits = model(batch)
        last = logits[mx.arange(len(active)), mx.array([len(seqs[i]) - 1 for i in active])]
        next_ids = mx.argmax(last[:, :n_vocab], axis=-1).tolist()
        for i, t in zip(active, next_ids):
            seqs[i].append(t)
            if t == eos or len(seqs[i]) - starts[i] >= max_new:
                done[i] = True
            elif t == result and store is not None:
                seqs[i] += tok.encode(store_reply(tok.decode(seqs[i]), store))
    return [tok.decode(s[start:]) for s, start in zip(seqs, starts)]
