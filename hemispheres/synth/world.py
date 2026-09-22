"""Synthetic worlds: entities, facts, and multi-hop traversal.

A world is a typed knowledge graph (people, companies, universities, cities,
countries) whose every fact is functional: each (subject, relation) has exactly
one object. Worlds generated with different `index` values never share an
entity name, but they share all tokens (see schema.py).
"""

import bisect
import hashlib
import random
from dataclasses import dataclass

from .schema import (CODAS, COMPANY_SUFFIXES, ENTITY_TYPES, RELATIONS, SYLLABLES, TRIVIAL_PAIRS,
                     UNIVERSITY_SUFFIXES, VALUE_DOMAINS, YEAR_RANGES)

# Entity names are partitioned into this many disjoint pools; world `index`
# draws only from pool `index`.
N_PARTITIONS = 8


@dataclass(frozen=True)
class WorldSizes:
    persons: int = 10_000
    companies: int = 800
    universities: int = 60
    cities: int = 200
    countries: int = 25


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: tuple[str, ...]


Hop = tuple[str, str, str]  # (subject id, relation, object id or value token)


class World:
    def __init__(self, name: str, index: int, seed: int, entities: list[Entity],
                 facts: dict[tuple[str, str], str]):
        self.name, self.index, self.seed = name, index, seed
        self.entities = {e.id: e for e in entities}
        self.facts = dict(facts)
        self._by_type = {t: [e for e in entities if e.type == t] for t in ENTITY_TYPES}
        self._by_name = {e.name: e.id for e in entities}
        assert len(self._by_name) == len(entities), "entity names must be unique within a world"

    def of_type(self, type_: str) -> list[Entity]:
        return self._by_type[type_]

    def surface(self, obj: str) -> tuple[str, ...]:
        """Tokens for an entity id or a value token."""
        e = self.entities.get(obj)
        return e.name if e else (obj,)

    def chain(self, subject: str, path: tuple[str, ...]) -> list[Hop] | None:
        """The hops followed from `subject` along `path`, or None if a fact is missing."""
        hops, cur = [], subject
        for rel in path:
            obj = self.facts.get((cur, rel))
            if obj is None:
                return None
            hops.append((cur, rel, obj))
            cur = obj
        return hops

    def answer(self, subject: str, path: tuple[str, ...]) -> str | None:
        hops = self.chain(subject, path)
        return hops[-1][2] if hops else None

    def lookup(self, name: tuple[str, ...], relation: str) -> tuple[str, ...] | None:
        """The store interface: surface name + relation -> surface value, or None."""
        subject = self._by_name.get(tuple(name))
        obj = self.facts.get((subject, relation)) if subject else None
        return self.surface(obj) if obj is not None else None

    def facts_of(self, subject: str) -> list[tuple[str, str]]:
        """(relation, object) pairs for a subject, in schema order."""
        return [(r, self.facts[(subject, r)]) for r in RELATIONS if (subject, r) in self.facts]

    def with_facts(self, updates: dict[tuple[str, str], str]) -> "World":
        return World(self.name, self.index, self.seed, list(self.entities.values()), {**self.facts, **updates})

    def to_json(self) -> dict:
        return {
            "name": self.name, "index": self.index, "seed": self.seed,
            "entities": [[e.id, e.type, list(e.name)] for e in self.entities.values()],
            "facts": [[s, r, o] for (s, r), o in self.facts.items()],
        }

    @classmethod
    def from_json(cls, d: dict) -> "World":
        return cls(d["name"], d["index"], d["seed"],
                   [Entity(i, t, tuple(n)) for i, t, n in d["entities"]],
                   {(s, r): o for s, r, o in d["facts"]})


def enumerate_paths(start: str = "person", max_hops: int = 3) -> list[tuple[str, ...]]:
    """All relation paths from an entity type, up to `max_hops`, in a stable order."""
    paths, frontier = [], [((), start)]
    for _ in range(max_hops):
        nxt = []
        for path, type_ in frontier:
            for rel in RELATIONS.values():
                if rel.subject != type_ or (path and (path[-1], rel.name) in TRIVIAL_PAIRS):
                    continue
                new = path + (rel.name,)
                paths.append(new)
                if rel.object_is_entity:
                    nxt.append((new, rel.object))
        frontier = nxt
    return paths


# --------------------------------------------------------------------------- generation


class _Namer:
    """Draws unique pseudo-word names from this world's partition of name space."""

    def __init__(self, rng: random.Random, index: int):
        self.rng, self.index, self.used = rng, index, set()

    def word(self, min_syl: int, max_syl: int, coda_p: float) -> tuple[str, ...]:
        syl = [self.rng.choice(SYLLABLES) for _ in range(self.rng.randint(min_syl, max_syl))]
        toks = [syl[0].capitalize()] + ["##" + s for s in syl[1:]]
        if self.rng.random() < coda_p:
            toks.append("##" + self.rng.choice(CODAS))
        return tuple(toks)

    def unique(self, make) -> tuple[str, ...]:
        while True:
            name = make()
            if name not in self.used and name_partition(name) == self.index:
                self.used.add(name)
                return name


def name_partition(name: tuple[str, ...]) -> int:
    digest = hashlib.md5(" ".join(name).encode()).digest()
    return int.from_bytes(digest[:8], "little") % N_PARTITIONS


def _zipf_sampler(rng: random.Random, items: list, a: float):
    """Sample items with probability ∝ 1/rank^a over a random ranking (a=0 is uniform)."""
    ranked = items[:]
    rng.shuffle(ranked)
    cum, total = [], 0.0
    for r in range(len(ranked)):
        total += 1.0 / (r + 1) ** a
        cum.append(total)
    return lambda: rng.choices(ranked, cum_weights=cum)[0]


def _year(rng: random.Random, relation: str) -> str:
    lo, hi = YEAR_RANGES[relation]
    return str(rng.randint(lo, hi))


def generate_world(name: str, index: int, seed: int = 0, sizes: WorldSizes = WorldSizes(),
                   zipf: float = 0.8) -> World:
    """Generate a world. Same (index, seed, sizes, zipf) always gives the same world."""
    if not 0 <= index < N_PARTITIONS:
        raise ValueError(f"index must be in [0, {N_PARTITIONS})")
    if sizes.cities < sizes.countries:
        raise ValueError("need at least one city per country")
    rng = random.Random(f"{seed}:{index}")
    namer = _Namer(rng, index)
    ents: dict[str, list[Entity]] = {t: [] for t in ENTITY_TYPES}
    facts: dict[tuple[str, str], str] = {}

    def make(type_, n, name_fn):
        ents[type_] = [Entity(f"{type_}:{i}", type_, namer.unique(name_fn)) for i in range(n)]
        return [e.id for e in ents[type_]]

    countries = make("country", sizes.countries, lambda: namer.word(2, 3, 0.5))
    cities = make("city", sizes.cities, lambda: namer.word(2, 3, 0.4))
    universities = make("university", sizes.universities,
                        lambda: namer.word(2, 2, 0.3) + (rng.choice(UNIVERSITY_SUFFIXES),))
    companies = make("company", sizes.companies, lambda: namer.word(2, 2, 0.3) + (rng.choice(COMPANY_SUFFIXES),))
    persons = make("person", sizes.persons, lambda: namer.word(2, 2, 0.3) + namer.word(2, 3, 0.5))

    # Geography: every country gets at least one city; its capital is one of its own cities.
    cities_in: dict[str, list[str]] = {c: [] for c in countries}
    for i, city in enumerate(cities):
        country = countries[i] if i < len(countries) else rng.choice(countries)
        facts[(city, "country")] = country
        cities_in[country].append(city)
    for country in countries:
        facts[(country, "capital")] = rng.choice(cities_in[country])
        facts[(country, "currency")] = rng.choice(VALUE_DOMAINS["currency"])

    popular_city = _zipf_sampler(rng, cities, zipf)
    for u in universities:
        facts[(u, "uni_city")] = popular_city()
    for c in companies:
        facts[(c, "hq")] = popular_city()
        facts[(c, "industry")] = rng.choice(VALUE_DOMAINS["industry"])
        facts[(c, "founded")] = _year(rng, "founded")

    popular_uni = _zipf_sampler(rng, universities, zipf)
    popular_company = _zipf_sampler(rng, companies, zipf)
    for p in persons:
        facts[(p, "born_in")] = popular_city()
        facts[(p, "birth_year")] = _year(rng, "birth_year")
        facts[(p, "studied_at")] = popular_uni()
        facts[(p, "major")] = rng.choice(VALUE_DOMAINS["major"])
        facts[(p, "works_at")] = popular_company()

    # Mentors are strictly older, so mentor chains never cycle. The oldest cohort has none.
    by_year = sorted(persons, key=lambda p: int(facts[(p, "birth_year")]))
    years = [int(facts[(p, "birth_year")]) for p in by_year]
    for p in persons:
        n_older = bisect.bisect_left(years, int(facts[(p, "birth_year")]))
        if n_older:
            facts[(p, "mentor")] = by_year[rng.randrange(n_older)]

    return World(name, index, seed, [e for t in ENTITY_TYPES for e in ents[t]], facts)
