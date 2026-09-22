"""Schema shared by every synthetic world: entity types, relations, value domains,
name pieces, templates and the vocabulary.

Worlds differ only in which entities exist and which facts hold. Every token a
world can produce comes from this module, so a model trained on one world has
seen every token another world uses. Only the combinations (entity names and
facts) are new.
"""

from dataclasses import dataclass

ENTITY_TYPES = ("person", "company", "university", "city", "country")

# Special tokens. Store-supplied spans are `[RESULT] value [END]`, and the
# `value [END]` part is excluded from the loss in the lookup format.
PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"
LOOKUP, RESULT, END = "[LOOKUP]", "[RESULT]", "[END]"
FACTS = "[FACTS]"
SPECIALS = (PAD, BOS, EOS, LOOKUP, RESULT, END, FACTS)


@dataclass(frozen=True)
class Relation:
    name: str
    subject: str                # an entity type
    object: str                 # an entity type, or a key of VALUE_DOMAINS
    phrases: tuple[str, ...]    # noun phrases for questions; "{s}" is the subject
    sentences: tuple[str, ...]  # statements for bios; "{s}" subject, "{o}" object

    @property
    def token(self) -> str:
        return "@" + self.name

    @property
    def object_is_entity(self) -> bool:
        return self.object in ENTITY_TYPES


RELATIONS: dict[str, Relation] = {r.name: r for r in (
    # person
    Relation("born_in", "person", "city",
             ("the birthplace of {s}", "the city where {s} was born"),
             ("{s} was born in {o} .", "{s} is a native of {o} .", "The birthplace of {s} is {o} .")),
    Relation("birth_year", "person", "year",
             ("the birth year of {s}", "the year {s} was born"),
             ("{s} was born in the year {o} .", "{s} came into the world in {o} .", "The birth year of {s} is {o} .")),
    Relation("studied_at", "person", "university",
             ("the university of {s}", "the university where {s} studied"),
             ("{s} studied at {o} .", "{s} graduated from {o} .", "{s} is an alumnus of {o} .")),
    Relation("major", "person", "major",
             ("the major of {s}", "the field {s} studied"),
             ("{s} majored in {o} .", "{s} earned a degree in {o} .", "The field of study of {s} was {o} .")),
    Relation("works_at", "person", "company",
             ("the employer of {s}", "the company where {s} works"),
             ("{s} works at {o} .", "{s} is employed by {o} .", "{s} has a job at {o} .")),
    Relation("mentor", "person", "person",
             ("the mentor of {s}", "the person who mentored {s}"),
             ("{s} was mentored by {o} .", "{o} is the mentor of {s} .", "{s} learned the trade from {o} .")),
    # company
    Relation("hq", "company", "city",
             ("the headquarters of {s}", "the city where {s} is headquartered"),
             ("{s} is headquartered in {o} .", "{s} has its head office in {o} .",
              "The headquarters of {s} is in {o} .")),
    Relation("industry", "company", "industry",
             ("the industry of {s}", "the sector of {s}"),
             ("{s} is a company in {o} .", "{s} operates in {o} .", "The sector of {s} is {o} .")),
    Relation("founded", "company", "year",
             ("the founding year of {s}", "the year {s} was founded"),
             ("{s} was founded in {o} .", "{s} opened for business in {o} .", "The founding year of {s} is {o} .")),
    # university
    Relation("uni_city", "university", "city",
             ("the location of {s}", "the city where {s} is located"),
             ("{s} is located in {o} .", "{s} has its campus in {o} .", "The campus of {s} is in {o} .")),
    # city
    Relation("country", "city", "country",
             ("the country of {s}", "the nation that {s} belongs to"),
             ("{s} is a city in {o} .", "{s} lies within {o} .", "{s} belongs to the nation of {o} .")),
    # country
    Relation("capital", "country", "city",
             ("the capital of {s}", "the capital city of {s}"),
             ("The capital of {s} is {o} .", "{o} is the capital city of {s} .", "{s} is governed from {o} .")),
    Relation("currency", "country", "currency",
             ("the currency of {s}", "the money used in {s}"),
             ("{s} uses the {o} as its currency .", "The currency of {s} is the {o} .",
              "People in {s} pay with the {o} .")),
)}

# Consecutive relation pairs that return to where they started (the country of a
# country's capital is that country); multi-hop paths never contain them.
TRIVIAL_PAIRS = {("capital", "country")}

QUESTION_FRAMES = ("Q: What is {x} ? A:", "Q: Name {x} . A:", "Q: Tell me {x} . A:")

VALUE_DOMAINS: dict[str, tuple[str, ...]] = {
    "year": tuple(str(y) for y in range(1850, 2011)),
    "major": ("physics", "chemistry", "biology", "mathematics", "economics", "history", "philosophy",
              "linguistics", "architecture", "medicine", "law", "music", "sculpture", "geology", "astronomy",
              "psychology", "sociology", "statistics", "engineering", "literature", "botany", "zoology",
              "theology", "finance"),
    "industry": ("shipping", "mining", "textiles", "software", "banking", "insurance", "aviation", "farming",
                 "publishing", "pharmaceuticals", "robotics", "logistics", "energy", "telecom", "retail",
                 "tourism", "brewing", "steel", "fishing", "cosmetics"),
    "currency": ("crown", "mark", "florin", "ducat", "shilling", "real", "dinar", "peso", "franc", "lira",
                 "thaler", "guilder", "rand", "kip", "sol", "lev"),
}

# Year ranges actually used when generating each year-valued relation.
YEAR_RANGES = {"birth_year": (1900, 1995), "founded": (1850, 2005)}

# Name pieces. A name is a word-initial syllable (capitalized) followed by
# continuation syllables ("##" prefix) and an optional coda, e.g.
# ("Ka", "##ro", "##n") -> "Karon".
_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "th", "sh", "st")
_VOWELS = ("a", "e", "i", "o", "u")
SYLLABLES = tuple(c + v for c in _ONSETS for v in _VOWELS)
CODAS = ("n", "r", "s", "l", "x", "th")
COMPANY_SUFFIXES = ("Industries", "Labs", "Systems", "Group", "Works", "Partners", "Dynamics", "Holdings",
                    "Trading", "Collective")
UNIVERSITY_SUFFIXES = ("University", "Institute", "College", "Academy")


def _template_words(template: str) -> list[str]:
    return [w for w in template.split() if not (w.startswith("{") and w.endswith("}"))]


def vocabulary() -> list[str]:
    """The fixed token list shared by all worlds, in a deterministic order."""
    out: list[str] = []
    seen: set[str] = set()

    def add(tokens):
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)

    add(SPECIALS)
    add(r.token for r in RELATIONS.values())
    for r in RELATIONS.values():
        for t in r.phrases + r.sentences:
            add(_template_words(t))
    for t in QUESTION_FRAMES:
        add(_template_words(t))
    for values in VALUE_DOMAINS.values():
        add(values)
    add(s.capitalize() for s in SYLLABLES)
    add("##" + s for s in SYLLABLES)
    add("##" + c for c in CODAS)
    add(COMPANY_SUFFIXES)
    add(UNIVERSITY_SUFFIXES)
    return out
