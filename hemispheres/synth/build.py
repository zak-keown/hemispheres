"""Build a synthetic world and its question splits, and write them to disk.

  python -m hemispheres.synth.build --name world-a --index 0
  python -m hemispheres.synth.build --name world-b --index 1 --edits 1,100,1000

Writes to data/<name>/:
  world.json    entities and facts
  vocab.json    the shared token list (identical for every world)
  splits.json   (subject, path) questions per split; rendering happens at load time
  edits/k<N>.json   nested counterfactual edit sets with direct/ripple/locality questions
  stats.json    counts
  samples.txt   one example of every rendering, for eyeballing

Splits (people are split into "compose" and "held-out" groups; see the
Grokked Transformers ID/OOD protocol):
  train          1-hop questions about every non-person entity and every compose
                 person, plus multi-hop questions about compose people
  test_id        held-out multi-hop questions about compose people
  test_ood       multi-hop questions about held-out people, who appear in no
                 multi-hop training question
  test_1hop_ood  1-hop questions about held-out people (knowledge extraction:
                 their facts appear only in bios / the store)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .. import provenance
from . import render
from .edits import edit_eval_set, sample_edits
from .schema import ENTITY_TYPES, RELATIONS, vocabulary
from .world import World, WorldSizes, enumerate_paths, generate_world

SPLITS = ("train", "test_id", "test_ood", "test_1hop_ood")


def make_splits(world: World, max_hops: int = 3, heldout_frac: float = 0.2, id_test_frac: float = 0.1,
                seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(f"splits:{seed}")
    persons = [p.id for p in world.of_type("person")]
    rng.shuffle(persons)
    heldout = set(persons[: int(len(persons) * heldout_frac)])
    paths = enumerate_paths("person", max_hops)
    splits: dict[str, list[tuple[str, str]]] = {s: [] for s in SPLITS}

    for type_ in ENTITY_TYPES:
        if type_ == "person":
            continue
        for e in world.of_type(type_):
            splits["train"] += [(e.id, rel) for rel, _ in world.facts_of(e.id)]

    for p in persons:
        for path in paths:
            if world.answer(p, path) is None:
                continue
            q = (p, "/".join(path))
            if p in heldout:
                splits["test_1hop_ood" if len(path) == 1 else "test_ood"].append(q)
            elif len(path) > 1 and rng.random() < id_test_frac:
                splits["test_id"].append(q)
            else:
                splits["train"].append(q)
    return splits


def samples_text(world: World, splits: dict, seed: int = 0) -> str:
    rng = random.Random(f"samples:{seed}")
    person = world.of_type("person")[0].id
    company = world.of_type("company")[0].id
    multi = next((s, p) for s, p in splits["train"] if s.startswith("person:") and p.count("/") == 2)
    subject, path = multi[0], tuple(multi[1].split("/"))
    blocks = [
        ("bio (dense)", render.bio(world, person, rng)),
        ("bio (dense, company)", render.bio(world, company, rng)),
        ("bio (lookup)", render.bio(world, person, rng, lookup=True)),
        ("qa direct", render.qa(world, subject, path, rng, "direct")),
        ("qa lookup", render.qa(world, subject, path, rng, "lookup")),
        ("qa context", render.qa(world, subject, path, rng, "context")),
    ]
    out = []
    for title, ex in blocks:
        trained = render.detokenize([t for t, m in zip(ex.tokens, ex.mask) if m])
        out += [f"== {title} ({len(ex.tokens)} tokens)", ex.text(), f"-- trained on: {trained}", ""]
    return "\n".join(out)


def stats(world: World, splits: dict, seed: int = 0) -> dict:
    rng = random.Random(f"stats:{seed}")
    by_hops = {s: dict(sorted(Counter(p.count("/") + 1 for _, p in qs).items())) for s, qs in splits.items()}
    persons = world.of_type("person")
    lengths = {}
    for style in ("direct", "lookup", "context"):
        qs = rng.sample(splits["train"], min(500, len(splits["train"])))
        lengths[f"qa_{style}"] = sum(len(render.qa(world, s, tuple(p.split("/")), rng, style).tokens)
                                     for s, p in qs) / len(qs)
    for lookup in (False, True):
        sample = rng.sample(persons, min(200, len(persons)))
        lengths[f"bio_person{'_lookup' if lookup else ''}"] = sum(
            len(render.bio(world, p.id, rng, lookup).tokens) for p in sample) / len(sample)
    return {
        "entities": {t: len(world.of_type(t)) for t in ENTITY_TYPES},
        "facts": len(world.facts),
        "facts_by_relation": dict(Counter(r for _, r in world.facts)),
        "paths_by_hops": dict(Counter(len(p) for p in enumerate_paths("person"))),
        "questions": {s: len(qs) for s, qs in splits.items()},
        "questions_by_hops": by_hops,
        "vocab_size": len(vocabulary()),
        "mean_tokens": {k: round(v, 1) for k, v in lengths.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--index", type=int, required=True, help="name partition; worlds with different indexes are disjoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--persons", type=int, default=WorldSizes.persons)
    p.add_argument("--companies", type=int, default=WorldSizes.companies)
    p.add_argument("--universities", type=int, default=WorldSizes.universities)
    p.add_argument("--cities", type=int, default=WorldSizes.cities)
    p.add_argument("--countries", type=int, default=WorldSizes.countries)
    p.add_argument("--zipf", type=float, default=0.8, help="popularity skew for cities, universities, employers")
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--id-test-frac", type=float, default=0.1)
    p.add_argument("--edits", default="", help="comma list of nested edit counts, e.g. 1,100,1000")
    p.add_argument("--edit-relations", default="",
                   help="comma list of relations to edit (default: all), e.g. works_at,mentor,hq")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    sizes = WorldSizes(args.persons, args.companies, args.universities, args.cities, args.countries)
    world = generate_world(args.name, args.index, args.seed, sizes, args.zipf)
    splits = make_splits(world, args.max_hops, args.heldout_frac, args.id_test_frac, args.seed)

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "world.json").write_text(json.dumps(world.to_json()))
    (out / "vocab.json").write_text(json.dumps(vocabulary()))
    (out / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in splits.items()}))
    (out / "samples.txt").write_text(samples_text(world, splits, args.seed))
    st = stats(world, splits, args.seed)
    st["config"] = vars(args)
    st["code"] = provenance.code_version()

    counts = sorted(int(k) for k in args.edits.split(",") if k)
    if counts:
        (out / "edits").mkdir(exist_ok=True)
        relations = tuple(r for r in args.edit_relations.split(",") if r) or None
        unknown = set(relations or ()) - set(RELATIONS)
        if unknown:
            p.error(f"unknown relations: {', '.join(sorted(unknown))}")
        edits = sample_edits(world, counts[-1], args.seed, relations)
        paths = enumerate_paths("person", args.max_hops)
        st["edits"] = {}
        for k in counts:
            ev = edit_eval_set(world, edits[:k], paths, args.seed)
            (out / "edits" / f"k{k}.json").write_text(json.dumps(ev))
            st["edits"][k] = {"direct": len(ev["direct"]), "ripple": len(ev["ripple"]),
                              "ripple_total": ev["n_ripple_total"], "locality": len(ev["locality"])}

    (out / "stats.json").write_text(json.dumps(st, indent=2))
    print(json.dumps({k: st[k] for k in ("entities", "facts", "questions", "vocab_size", "mean_tokens")
                      + (("edits",) if counts else ())}, indent=2))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
