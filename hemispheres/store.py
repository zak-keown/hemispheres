"""A world's facts as fixed-shape arrays for latent-store models.

Entry j holds one fact (s, r, o): the subject's name tokens and the relation
(from which the model computes the entry's key), and the object's name tokens
(the value the model reads). Writing a new world or applying edits means
building a new Store; no model weights change.

A store can be padded to a fixed `size` (so stores of different worlds share
array shapes). Padding entries have `bias` -1e9, which is added to their
retrieval scores so they are never retrieved.
"""

import mlx.core as mx

from .synth.render import Tokenizer
from .synth.schema import PAD, RELATIONS
from .synth.world import World

MAX_NAME_LEN = 8
RELATION_INDEX = {r: i for i, r in enumerate(RELATIONS)}


class Store:
    def __init__(self, world: World, tok: Tokenizer, size: int | None = None):
        self.facts = list(world.facts)
        self.index = {k: j for j, k in enumerate(self.facts)}
        pad = tok.token_id(PAD)
        n_pad = 0 if size is None else size - len(self.facts)
        if n_pad < 0:
            raise ValueError(f"store has {len(self.facts)} facts, more than size={size}")

        def padded(names):
            ids, mask = [], []
            for name in names:
                if len(name) > MAX_NAME_LEN:
                    raise ValueError(f"name {name} is longer than {MAX_NAME_LEN} tokens")
                ids.append(tok.encode(list(name)) + [pad] * (MAX_NAME_LEN - len(name)))
                mask.append([1.0] * len(name) + [0.0] * (MAX_NAME_LEN - len(name)))
            return mx.array(ids, dtype=mx.int32), mx.array(mask)

        subj, subj_mask = padded([world.surface(s) for s, _ in self.facts] + [()] * n_pad)
        val, val_mask = padded([world.surface(world.facts[k]) for k in self.facts] + [()] * n_pad)
        self.arrays = {
            "subj": subj, "subj_mask": subj_mask,
            "rel": mx.array([RELATION_INDEX[r] for _, r in self.facts] + [0] * n_pad, dtype=mx.int32),
            "val": val, "val_mask": val_mask,
            "bias": mx.array([0.0] * len(self.facts) + [-1e9] * n_pad),
        }

    def __len__(self) -> int:
        return len(self.facts)
