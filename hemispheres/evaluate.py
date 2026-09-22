"""Evaluate a trained model by exact match on generated answers.

  # Held-out questions in the training world
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a
  # World swap: a world the model never saw
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-b --sets all
  # Counterfactual edits: direct / ripple / locality questions after 100 edits
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a --edits 100

An answer is correct only if the generated text, after the last store reply in
the lookup format, is exactly the answer followed by [EOS]. For the lookup arm
the store is the evaluated world (with edits applied). The context arm gets that
world's facts in its prompt. The dense arm gets nothing, so it must have been
updated (see train.py --init) to know a new world or edits.
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import checkpoint
from .data import ARMS, WorldData
from .generate import generate
from .synth import render
from .synth.render import Tokenizer
from .synth.schema import END, EOS, LOOKUP, RESULT

SPLIT_SETS = ("test_id", "test_ood", "test_1hop_ood", "train", "all")
EDIT_SETS = ("direct", "ripple", "locality")


@dataclass
class Question:
    set: str
    subject: str
    path: tuple[str, ...]
    answer: str
    meta: dict = field(default_factory=dict)


def split_questions(data: WorldData, split: str, n_per_hop: int, seed: int = 0) -> list[Question]:
    """Up to `n_per_hop` questions per hop count from a split ("all" = every split)."""
    qs = [q for qs in data.splits.values() for q in qs] if split == "all" else data.splits[split]
    by_hops = defaultdict(list)
    for subject, path in qs:
        by_hops[len(path)].append((subject, path))
    rng = random.Random(f"eval:{split}:{seed}")
    out = []
    for hops in sorted(by_hops):
        for subject, path in rng.sample(by_hops[hops], min(n_per_hop, len(by_hops[hops]))):
            out.append(Question(split, subject, path, data.world.answer(subject, path)))
    return out


def edit_questions(data: WorldData, kind: str, n: int, seed: int = 0) -> list[Question]:
    items = data.edit_set[kind]
    items = random.Random(f"eval:{kind}:{seed}").sample(items, min(n, len(items)))
    out = []
    for q in items:
        meta = {k: q[k] for k in ("changed", "edit_hop", "before") if k in q}
        out.append(Question(kind, q["subject"], tuple(q["path"].split("/")), q["answer"], meta))
    return out


def prompt_tokens(data: WorldData, q: Question, style: str, seed: int = 0) -> list[str]:
    rng = random.Random(f"prompt:{q.subject}:{'/'.join(q.path)}:{seed}")
    ex = render.qa(data.world, q.subject, q.path, rng, style)
    return ex.tokens[:ex.tokens.index("A:") + 1]


def parse_output(output: list[str], style: str) -> tuple[list[str] | None, list[tuple]]:
    """(final answer tokens or None if generation never ended, lookup queries made)."""
    queries, i = [], 0
    while LOOKUP in output[i:]:
        i = output.index(LOOKUP, i) + 1
        if RESULT not in output[i:]:
            break
        j = output.index(RESULT, i)
        queries.append((tuple(output[i:j - 1]), output[j - 1].lstrip("@")))
        i = j
    if EOS not in output:
        return None, queries
    body = output[:output.index(EOS)]
    if style == "lookup" and END in body:
        body = body[len(body) - body[::-1].index(END):]
    return body, queries


def score(model, tok: Tokenizer, data: WorldData, questions: list[Question], style: str, seed: int = 0,
          batch_size: int = 256) -> list[dict]:
    prompts = [prompt_tokens(data, q, style, seed) for q in questions]
    max_new = 24 + 16 * max((len(q.path) for q in questions), default=1) if style == "lookup" else 24
    outputs = generate(model, tok, prompts, max_new=max_new, store=data.world if style == "lookup" else None,
                       batch_size=batch_size)
    records = []
    for q, out in zip(questions, outputs):
        final, queries = parse_output(out, style)
        gold = list(data.world.surface(q.answer))
        rec = {"set": q.set, "subject": q.subject, "path": "/".join(q.path), "hops": len(q.path),
               "answer": render.detokenize(gold), "output": render.detokenize(out),
               "correct": final == gold, **q.meta}
        if style == "lookup":
            chain = data.world.chain(q.subject, q.path)
            rec["trace_ok"] = queries == [(data.world.surface(s), r) for s, r, _ in chain]
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["set"], f"{r['hops']}hop")].append(r)
        if r["set"] == "ripple":
            groups[("ripple", "changed" if r.get("changed") else "unchanged")].append(r)
            groups[("ripple", f"edit_hop{r.get('edit_hop')}")].append(r)
        groups[(r["set"], "all")].append(r)
    out: dict = defaultdict(dict)
    for (s, key), rs in sorted(groups.items()):
        out[s][key] = {"acc": sum(r["correct"] for r in rs) / len(rs), "n": len(rs)}
        if "trace_ok" in rs[0]:
            out[s][key]["trace_ok"] = sum(r["trace_ok"] for r in rs) / len(rs)
    return dict(out)


def format_summary(summary: dict) -> str:
    lines = []
    for s, groups in summary.items():
        cells = []
        for key, v in groups.items():
            extra = f", trace {v['trace_ok']:.0%}" if "trace_ok" in v else ""
            cells.append(f"{key} {v['acc']:.1%} (n={v['n']}{extra})")
        lines.append(f"{s:>14}: " + " · ".join(cells))
    return "\n".join(lines)


def evaluate(model, tok: Tokenizer, data: WorldData, arm: str, sets: list[str], n: int, seed: int = 0) -> tuple:
    questions = []
    for s in sets:
        questions += edit_questions(data, s, n, seed) if s in EDIT_SETS else split_questions(data, s, n, seed)
    records = score(model, tok, data, questions, ARMS[arm].qa_style, seed)
    return summarize(records), records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--data", required=True, help="built world directory")
    p.add_argument("--edits", type=int, default=0, help="evaluate with this many edits applied")
    p.add_argument("--sets", default=None,
                   help=f"comma list from {SPLIT_SETS + EDIT_SETS} (default: test splits, or edit sets with --edits)")
    p.add_argument("--n", type=int, default=500, help="questions per hop count (splits) or per set (edits)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="output name under <run>/evals/")
    args = p.parse_args()

    model, config = checkpoint.load_model(args.run, args.checkpoint)
    data = WorldData(args.data, args.edits)
    tok = Tokenizer(data.vocab)
    sets = (args.sets.split(",") if args.sets else
            list(EDIT_SETS) if args.edits else ["test_id", "test_ood", "test_1hop_ood"])
    summary, records = evaluate(model, tok, data, config["arm"], sets, args.n, args.seed)
    print(f"{config['arm']} · {args.run} [{args.checkpoint}] on {data.path.name}"
          + (f" with {args.edits} edits" if args.edits else ""))
    print(format_summary(summary))

    name = args.name or f"{data.path.name}{f'-k{args.edits}' if args.edits else ''}-{args.checkpoint}"
    out = Path(args.run) / "evals"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2))
    with open(out / f"{name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
