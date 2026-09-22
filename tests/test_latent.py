import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from hemispheres.data import ExampleSampler, Packer, WorldData, parse_mix
from hemispheres.evaluate import evaluate
from hemispheres.latent import LatentConfig, LatentGPT, latent_loss
from hemispheres.store import Store
from hemispheres.synth import Tokenizer, WorldSizes, generate_world, vocabulary
from hemispheres.synth.build import make_splits


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    w = generate_world("t", 0, sizes=WorldSizes(persons=200, companies=20, universities=5, cities=12, countries=3))
    (d / "world.json").write_text(json.dumps(w.to_json()))
    (d / "vocab.json").write_text(json.dumps(vocabulary()))
    (d / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in make_splits(w).items()}))
    return WorldData(d)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    return LatentGPT(LatentConfig(n_layer=4, n_head=4, d_model=64, vocab_size=576, n_reads=3, top_k=3, key_dim=32,
                                  retrieval_chunk=64))


def test_store_entries_match_the_world(data):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    assert len(store) == len(data.world.facts)
    (s, r), j = next(iter(store.index.items()))
    val = store.arrays["val"][j].tolist()[:int(store.arrays["val_mask"][j].sum().item())]
    assert tuple(tok.decode(val)) == data.world.surface(data.world.facts[(s, r)])


def test_read_layers_are_spread_through_the_stack():
    assert LatentConfig(n_layer=8, n_reads=3).read_after == [2, 4, 6]
    assert LatentConfig(n_layer=6, n_reads=3).read_after == [2, 3, 4]


def test_retrieval_is_exact_top_k(model):
    q = mx.random.normal((2, 5, 32))
    keys = mx.random.normal((300, 32))
    idx = model.reads[0].retrieve(q, keys)
    scores = (q.reshape(-1, 32) @ keys.T)
    for row, got in zip(scores.tolist(), idx.reshape(-1, 3).tolist()):
        best = sorted(range(len(row)), key=lambda j: -row[j])[:3]
        assert set(got) == set(best)


def test_forward_and_loss_train_queries_and_keys(data, model):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    packer = Packer(ExampleSampler(data, "latent", parse_mix(None, "latent")), tok, batch_size=2, seq_len=96,
                    store=store, n_reads=3, max_supervision=32)
    batch = packer.batch()
    logits, queries, keys = model.forward(batch[0], store.arrays)
    assert logits.shape == (2, 96, 576) and len(queries) == 3 and keys.shape == (len(store), 32)

    def loss(*b):
        return latent_loss(model, store.arrays, *b, hop_weight=1.0)[0]

    _, grads = nn.value_and_grad(model, loss)(*batch)
    g = dict(tree_flatten(grads))
    for name in ("key_encoder.out.weight", "reads.0.query.weight", "reads.2.query.weight", "reads.1.k.weight"):
        assert mx.abs(g[name]).sum().item() > 0, name


def _supervised(data, mix):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    packer = Packer(ExampleSampler(data, "latent", parse_mix(mix, "latent"), seed=3), tok, batch_size=4,
                    seq_len=128, store=store, n_reads=3, max_supervision=512)
    _, targets, _, sup_idx, sup_fact, sup_valid = packer.batch()
    slots = [(h, r, p, f) for (h, r, p), f, v in zip(sup_idx.tolist(), sup_fact.tolist(), sup_valid.tolist()) if v]
    assert slots
    return tok, store, targets.tolist(), slots


def test_bio_supervision_points_at_the_object_about_to_be_written(data):
    tok, store, targets, slots = _supervised(data, "bios=1")
    for hop, row, pos, fact in slots:
        assert hop == 0
        obj = data.world.facts[store.facts[fact]]
        assert tok.vocab[targets[row][pos]] == data.world.surface(obj)[0]


def test_qa_supervision_is_one_fact_per_hop_ending_at_the_answer(data):
    tok, store, targets, slots = _supervised(data, "qa=1")
    by_pos = {}
    for hop, row, pos, fact in slots:
        by_pos.setdefault((row, pos), {})[hop] = fact
    for (row, pos), hops in by_pos.items():
        assert sorted(hops) == list(range(len(hops)))
        chain = [store.facts[hops[h]] for h in sorted(hops)]
        for (s1, r1), (s2, _) in zip(chain, chain[1:]):
            assert data.world.facts[(s1, r1)] == s2  # each hop starts where the last one ended
        answer = data.world.facts[chain[-1]]
        assert tok.vocab[targets[row][pos]] == data.world.surface(answer)[0]


def test_untrained_latent_model_evaluates(data, model):
    summary, records = evaluate(model, Tokenizer(data.vocab), data, "latent", ["test_id"], n=3)
    assert records and "test_id" in summary
