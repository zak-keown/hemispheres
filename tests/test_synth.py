import random

import pytest

from hemispheres.synth import (RELATIONS, Tokenizer, World, WorldSizes, apply_edits, bio, edit_eval_set,
                               enumerate_paths, generate_world, qa, sample_edits, vocabulary)
from hemispheres.synth.build import make_splits
from hemispheres.synth.schema import END, RESULT

SIZES = WorldSizes(persons=600, companies=60, universities=12, cities=30, countries=6)


@pytest.fixture(scope="module")
def world_a():
    return generate_world("a", 0, seed=0, sizes=SIZES)


@pytest.fixture(scope="module")
def world_b():
    return generate_world("b", 1, seed=0, sizes=SIZES)


def test_deterministic(world_a):
    assert generate_world("a", 0, seed=0, sizes=SIZES).to_json() == world_a.to_json()
    assert generate_world("a", 0, seed=1, sizes=SIZES).to_json() != world_a.to_json()


def test_worlds_share_no_entity_names(world_a, world_b):
    names_a = {e.name for e in world_a.entities.values()}
    names_b = {e.name for e in world_b.entities.values()}
    assert not names_a & names_b


def test_every_rendered_token_is_in_the_shared_vocab(world_a, world_b):
    tok = Tokenizer.default()
    rng = random.Random(0)
    for world in (world_a, world_b):
        for e in list(world.entities.values())[::7]:
            for lookup in (False, True):
                tok.encode(bio(world, e.id, rng, lookup).tokens)
        for p in world.of_type("person")[:50]:
            for path in enumerate_paths():
                if world.answer(p.id, path) is not None:
                    for style in ("direct", "lookup", "context"):
                        tok.encode(qa(world, p.id, path, rng, style).tokens)
    assert len(vocabulary()) == len(set(vocabulary()))


def test_world_constraints(world_a):
    f = world_a.facts
    for c in world_a.of_type("country"):
        assert f[(f[(c.id, "capital")], "country")] == c.id
    for p in world_a.of_type("person"):
        if (p.id, "mentor") in f:
            assert int(f[(f[(p.id, "mentor")], "birth_year")]) < int(f[(p.id, "birth_year")])
    for (s, r), o in f.items():
        rel = RELATIONS[r]
        assert world_a.entities[s].type == rel.subject
        if rel.object_is_entity:
            assert world_a.entities[o].type == rel.object


def test_paths_are_well_typed_and_nontrivial():
    for path in enumerate_paths(max_hops=3):
        assert RELATIONS[path[0]].subject == "person"
        for a, b in zip(path, path[1:]):
            assert RELATIONS[a].object == RELATIONS[b].subject
            assert (a, b) != ("capital", "country")


def test_json_round_trip(world_a):
    assert World.from_json(world_a.to_json()).to_json() == world_a.to_json()


def test_lookup_trace_matches_the_chain_and_masks_store_tokens(world_a):
    rng = random.Random(0)
    p = world_a.of_type("person")[3].id
    path = ("works_at", "hq", "country")
    ex = qa(world_a, p, path, rng, "lookup")
    hops = world_a.chain(p, path)
    # Each [RESULT] ... [END] span holds the next hop's object and is excluded from the loss.
    starts = [i for i, t in enumerate(ex.tokens) if t == RESULT]
    assert len(starts) == len(hops)
    for i, (_, _, obj) in zip(starts, hops):
        end = ex.tokens.index(END, i)
        assert tuple(ex.tokens[i + 1:end]) == world_a.surface(obj)
        assert all(m == 0 for m in ex.mask[i + 1:end + 1])
    # The store interface returns what the trace shows.
    s, r, o = hops[1]
    assert world_a.lookup(world_a.surface(s), r) == world_a.surface(o)


def test_qa_trains_only_on_the_answer(world_a):
    rng = random.Random(0)
    p = world_a.of_type("person")[5].id
    for style in ("direct", "context"):
        ex = qa(world_a, p, ("born_in", "country"), rng, style)
        trained = [t for t, m in zip(ex.tokens, ex.mask) if m]
        assert tuple(trained[:-1]) == world_a.surface(world_a.answer(p, ("born_in", "country")))


def test_context_style_contains_the_gold_facts(world_a):
    rng = random.Random(0)
    p = world_a.of_type("person")[8].id
    path = ("studied_at", "uni_city", "country")
    ex = qa(world_a, p, path, rng, "context")
    prompt = " ".join(ex.tokens)
    for s, _, o in world_a.chain(p, path):
        assert " ".join(world_a.surface(s)) in prompt and " ".join(world_a.surface(o)) in prompt


def test_splits_keep_heldout_people_out_of_multihop_training(world_a):
    splits = make_splits(world_a, heldout_frac=0.2, id_test_frac=0.1)
    heldout = {s for s, _ in splits["test_ood"]} | {s for s, _ in splits["test_1hop_ood"]}
    train = set(map(tuple, splits["train"]))
    assert heldout and not heldout & {s for s, _ in train}
    assert not train & set(map(tuple, splits["test_id"]))
    assert all(p.count("/") >= 1 for _, p in splits["test_id"] + splits["test_ood"])


def test_edits_are_consistent_and_nested(world_a):
    edits = sample_edits(world_a, 200, seed=0)
    assert len({(e.subject, e.relation) for e in edits}) == 200
    assert all(e.old != e.new for e in edits)
    after = apply_edits(world_a, edits)
    for c in after.of_type("country"):  # every prefix keeps capitals inside their country
        assert after.facts[(after.facts[(c.id, "capital")], "country")] == c.id
    assert sample_edits(world_a, 50, seed=0) == edits[:50]


def test_edit_eval_sets(world_a):
    edits = sample_edits(world_a, 100, seed=0)
    after = apply_edits(world_a, edits)
    edited = {(e.subject, e.relation) for e in edits}
    ev = edit_eval_set(world_a, edits, enumerate_paths(), n_locality=300)
    for q in ev["direct"]:
        assert after.answer(q["subject"], (q["path"],)) == q["answer"] != q["before"]
    assert ev["ripple"]
    for q in ev["ripple"]:
        path = tuple(q["path"].split("/"))
        hops = after.chain(q["subject"], path)
        assert hops[-1][2] == q["answer"] and len(path) >= 2
        assert (hops[q["edit_hop"]][0], hops[q["edit_hop"]][1]) in edited
        assert q["changed"] == (world_a.answer(q["subject"], path) != q["answer"])
    for q in ev["locality"]:
        path = tuple(q["path"].split("/"))
        assert world_a.answer(q["subject"], path) == after.answer(q["subject"], path) == q["answer"]
