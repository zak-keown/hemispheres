"""Counterfactual edits and the question sets that measure them.

An edit changes the object of one (subject, relation) fact. For a set of edits
we build three question sets, all answered in the edited world:

  direct    the edited fact itself (1 hop from the edited subject)
  ripple    multi-hop questions from people whose path passes through an edited
            fact (the propagation test; `changed` marks answers that differ from
            the original world)
  locality  questions that touch no edited fact (their answers must not move)

Edits are sampled once as a list and evaluated on prefixes of it, so results
for 1, 100 and 1,000 edits are nested.
"""

import random
from dataclasses import asdict, dataclass

from .schema import RELATIONS, VALUE_DOMAINS, YEAR_RANGES
from .world import World


@dataclass(frozen=True)
class Edit:
    subject: str
    relation: str
    old: str
    new: str


def _alternative(world: World, facts: dict, subject: str, relation: str, rng: random.Random) -> str | None:
    """A new object that keeps the world's constraints under the current `facts`, or None."""
    rel, old = RELATIONS[relation], facts[(subject, relation)]
    if relation == "mentor":
        year = int(facts[(subject, "birth_year")])
        pool = [p.id for p in world.of_type("person")
                if p.id != subject and int(facts[(p.id, "birth_year")]) < year]
    elif relation == "capital":
        pool = [c.id for c in world.of_type("city") if facts[(c.id, "country")] == subject]
    elif relation == "country":
        if facts[(old, "capital")] == subject:
            return None  # moving a capital would leave its country without one
        pool = [c.id for c in world.of_type("country")]
    elif rel.object_is_entity:
        pool = [e.id for e in world.of_type(rel.object)]
    elif relation in YEAR_RANGES:
        lo, hi = YEAR_RANGES[relation]
        pool = [str(y) for y in range(lo, hi + 1)]
    else:
        pool = list(VALUE_DOMAINS[rel.object])
    pool = [o for o in pool if o != old]
    return rng.choice(pool) if pool else None


def sample_edits(world: World, n: int, seed: int = 0, relations: tuple[str, ...] | None = None) -> list[Edit]:
    """`n` edits to distinct facts, in a random order (use prefixes for smaller counts).

    Each edit is sampled against the world with all earlier edits applied, so every
    prefix of the list is a consistent world.
    """
    rng = random.Random(f"edits:{seed}")
    keys = [k for k in world.facts if relations is None or k[1] in relations]
    rng.shuffle(keys)
    facts, edits = dict(world.facts), []
    for subject, relation in keys:
        if len(edits) == n:
            break
        new = _alternative(world, facts, subject, relation, rng)
        if new is not None:
            edits.append(Edit(subject, relation, facts[(subject, relation)], new))
            facts[(subject, relation)] = new
    if len(edits) < n:
        raise ValueError(f"only {len(edits)} editable facts, asked for {n}")
    return edits


def apply_edits(world: World, edits: list[Edit]) -> World:
    return world.with_facts({(e.subject, e.relation): e.new for e in edits})


def edit_eval_set(world: World, edits: list[Edit], paths: list[tuple[str, ...]], seed: int = 0,
                  max_ripple: int = 2000, n_locality: int = 1000) -> dict:
    """Question sets for `edits` applied together to `world`. Paths start from people."""
    rng = random.Random(f"edit-eval:{seed}:{len(edits)}")
    after = apply_edits(world, edits)
    edited = {(e.subject, e.relation) for e in edits}

    direct = [{"subject": e.subject, "path": e.relation, "answer": e.new, "before": e.old} for e in edits]
    ripple, untouched = [], []
    for p in world.of_type("person"):
        for path in paths:
            hops = after.chain(p.id, path)
            if hops is None:
                continue
            edit_hops = [i for i, (s, r, _) in enumerate(hops) if (s, r) in edited]
            if edit_hops and len(path) >= 2:
                before = world.answer(p.id, path)
                ripple.append({"subject": p.id, "path": "/".join(path), "answer": hops[-1][2], "before": before,
                               "changed": before != hops[-1][2], "edit_hop": edit_hops[0], "hops": len(path)})
            elif not edit_hops:
                untouched.append({"subject": p.id, "path": "/".join(path), "answer": hops[-1][2],
                                  "hops": len(path)})

    # Keep every ripple question whose answer changed before any whose answer did not.
    rng.shuffle(ripple)
    ripple.sort(key=lambda q: not q["changed"])
    return {
        "n_edits": len(edits),
        "edits": [asdict(e) for e in edits],
        "direct": direct,
        "ripple": ripple[:max_ripple],
        "n_ripple_total": len(ripple),
        "locality": rng.sample(untouched, min(n_locality, len(untouched))),
    }
