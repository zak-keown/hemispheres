"""Render a world into training and evaluation examples.

The same facts can be rendered four ways, one per experimental arm:

  bio(..., lookup=False)    plain biographies: facts to be memorized (dense arm)
  bio(..., lookup=True)     biographies whose facts come from the store via inline
                            `[LOOKUP] subject @rel [RESULT] value [END]` calls (lookup arm)
  qa(..., style="direct")   question -> answer, no help (dense and latent-store arms)
  qa(..., style="lookup")   question -> one lookup per hop -> answer (lookup arm)
  qa(..., style="context")  gold facts + distractors in the prompt -> answer (in-context oracle)

Every example carries a per-token loss mask. Store-supplied tokens (`value [END]`
after `[RESULT]`) and prompts are masked out; the model is trained only on what
it must produce itself.

Examples also record retrieval supervision for latent-store models: at token
position `pos`, read layer `hop` should retrieve the fact (subject, relation).
Supervision covers every position that writes a token of the object (or
answer) name; `offset` is the index of the name token written next, so
offset 0 is the position just before the name starts.
"""

import random
from dataclasses import dataclass, field

from .schema import BOS, END, EOS, FACTS, LOOKUP, QUESTION_FRAMES, RELATIONS, RESULT, vocabulary
from .world import World


@dataclass
class Example:
    tokens: list[str] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)  # 1 = counts toward the loss
    meta: dict = field(default_factory=dict)
    supervision: list[tuple[int, int, str, str, int]] = field(default_factory=list)  # (pos, hop, subject, relation, offset)

    def add(self, tokens, trainable: bool = True) -> None:
        tokens = list(tokens)
        self.tokens += tokens
        self.mask += [int(trainable)] * len(tokens)

    def text(self) -> str:
        return detokenize(self.tokens)


class Tokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.ids = {t: i for i, t in enumerate(self.vocab)}

    @classmethod
    def default(cls) -> "Tokenizer":
        return cls(vocabulary())

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, tokens: list[str]) -> list[int]:
        try:
            return [self.ids[t] for t in tokens]
        except KeyError as e:
            raise KeyError(f"token {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: list[int]) -> list[str]:
        return [self.vocab[i] for i in ids]

    def token_id(self, token: str) -> int:
        return self.ids[token]


def detokenize(tokens: list[str]) -> str:
    """Human-readable text: glue `##` continuations and punctuation to the previous token."""
    out = ""
    for t in tokens:
        if t.startswith("##"):
            out += t[2:]
        elif t in (".", ",", "?") and out:
            out += t
        else:
            out += (" " if out else "") + t
    return out


def _fill(ex: Example, template: str, slots: dict, trainable: bool = True) -> None:
    for w in template.split():
        if w.startswith("{") and w.endswith("}"):
            slot = slots[w[1:-1]]
            slot(ex) if callable(slot) else ex.add(slot, trainable)
        else:
            ex.add([w], trainable)


def emit_lookup(ex: Example, subject: tuple[str, ...], relation: str, value: tuple[str, ...]) -> None:
    """A store call. The model writes the query; the store writes `value [END]`."""
    ex.add([LOOKUP, *subject, RELATIONS[relation].token, RESULT])
    ex.add([*value, END], trainable=False)


def statement(ex: Example, world: World, subject: str, relation: str, rng: random.Random,
              lookup: bool = False, trainable: bool = True) -> None:
    """One fact as a sentence, with a random template.

    When the subject precedes the object, a latent-store model should retrieve
    the fact at every position that writes an object token.
    """
    s, o = world.surface(subject), world.surface(world.facts[(subject, relation)])
    template = rng.choice(RELATIONS[relation].sentences)
    if lookup:
        _fill(ex, template, {"s": s, "o": lambda e: emit_lookup(e, s, relation, o)}, trainable)
        return

    def obj(e: Example) -> None:
        if "{s}" in template.split()[:template.split().index("{o}")]:
            e.supervision += [(len(e.tokens) - 1 + i, 0, subject, relation, i) for i in range(len(o))]
        e.add(o, trainable)

    _fill(ex, template, {"s": s, "o": obj}, trainable)


def bio(world: World, subject: str, rng: random.Random, lookup: bool = False) -> Example:
    """All facts about one entity, in random order with random templates."""
    facts = world.facts_of(subject)
    rng.shuffle(facts)
    ex = Example(meta={"kind": "bio", "subject": subject, "lookup": lookup})
    ex.add([BOS])
    for rel, _ in facts:
        statement(ex, world, subject, rel, rng, lookup=lookup)
    ex.add([EOS])
    return ex


def question_tokens(world: World, subject: str, path: tuple[str, ...], rng: random.Random) -> list[str]:
    """E.g. path (works_at, hq) -> "Q: What is the headquarters of the employer of <subject> ? A:"."""
    phrase = list(world.surface(subject))
    for rel in path:
        inner = Example()
        _fill(inner, rng.choice(RELATIONS[rel].phrases), {"s": phrase})
        phrase = inner.tokens
    q = Example()
    _fill(q, rng.choice(QUESTION_FRAMES), {"x": phrase})
    return q.tokens


def qa(world: World, subject: str, path: tuple[str, ...], rng: random.Random, style: str = "direct",
       n_distractors: int = 6) -> Example:
    hops = world.chain(subject, path)
    if hops is None:
        raise ValueError(f"{subject} has no answer for path {path}")
    answer = hops[-1][2]
    ex = Example(meta={"kind": "qa", "style": style, "subject": subject, "path": "/".join(path),
                       "hops": len(path), "answer": answer})
    ex.add([BOS], trainable=False)
    if style == "context":
        gold = [(s, r) for s, r, _ in hops]
        shown = gold + _distractors(world, gold, path, rng, n_distractors)
        rng.shuffle(shown)
        ex.add([FACTS], trainable=False)
        for s, r in shown:
            statement(ex, world, s, r, rng, trainable=False)
    elif style not in ("direct", "lookup"):
        raise ValueError(f"unknown style {style!r}")
    ex.add(question_tokens(world, subject, path, rng), trainable=False)
    start = len(ex.tokens) - 1  # the "A:" token, which predicts the first answer token
    ex.supervision += [(start + i, hop, s, r, i) for i in range(len(world.surface(answer)))
                       for hop, (s, r, _) in enumerate(hops)]
    if style == "lookup":
        for s, r, o in hops:
            emit_lookup(ex, world.surface(s), r, world.surface(o))
    ex.add([*world.surface(answer), EOS])
    return ex


def _distractors(world: World, gold: list[tuple[str, str]], path: tuple[str, ...], rng: random.Random,
                 n: int) -> list[tuple[str, str]]:
    """Facts that share a relation with the path, so the answer can't be found by relation alone."""
    gold_set, out = set(gold), []
    rels = list(dict.fromkeys(path))
    for _ in range(100 * n):
        if len(out) == n:
            break
        rel = rng.choice(rels)
        subject = rng.choice(world.of_type(RELATIONS[rel].subject)).id
        key = (subject, rel)
        if key in world.facts and key not in gold_set and key not in out:
            out.append(key)
    return out
