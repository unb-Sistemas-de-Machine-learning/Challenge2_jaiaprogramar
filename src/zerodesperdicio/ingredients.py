"""Normalização de linhas de ingrediente: "3 xícaras (chá) de açúcar" -> açúcar, 3, xícara."""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from .ingredient_rules import UNITS as _UNITS
from .ingredient_rules import (
    PREP_VERBS,
    PROTECTED,
    QUALIFIERS,
    SINGULAR_S,
    SYNONYMS,
    UNIT_SIZES,
    UNITS,
)


@dataclass(frozen=True)
class ParsedIngredient:
    raw: str
    name: str
    key: str
    quantity: float | None
    unit: str | None


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# "gotas de limão", "grande de polvilho": unidade ou tamanho que sobrou na frente do nome
_PREFIX_DROP = (frozenset(_UNITS) | QUALIFIERS | UNIT_SIZES) - {"molho", "molhos"}


def _singular(word: str) -> str:
    if len(word) <= 3 or word in SINGULAR_S or not word.endswith("s"):
        return word
    for suffix, repl in (("oes", "ao"), ("aes", "ao"), ("res", "r"), ("zes", "z"),
                         ("eis", "el"), ("ais", "al"), ("ois", "ol"), ("uis", "ul"), ("ns", "m")):
        if word.endswith(suffix):
            return word[: -len(suffix)] + repl
    if word.endswith(("ss", "us")):
        return word
    return word[:-1]


def ingredient_key(name: str) -> str:
    """Chave canônica: minúsculas, sem acento, singular, sem qualificadores, sinônimos aplicados."""
    text = strip_accents(name.lower()).replace("-", " ")
    words = re.findall(r"[a-z]+", text)
    if not words:
        return ""
    words = [_singular(w) for w in words]
    while len(words) > 2 and words[1] in {"de", "do", "da"} and words[0] in _PREFIX_DROP:
        words = words[2:]
    if "sem" in words[1:]:
        words = words[: words.index("sem", 1)]
    phrase = " ".join(words)
    if phrase not in PROTECTED and phrase not in SYNONYMS:
        words = [words[0]] + [w for w in words[1:] if w not in QUALIFIERS]
        while len(words) > 1 and words[-1] in {"de", "do", "da", "dos", "das", "e", "com"}:
            words.pop()
        phrase = " ".join(words)
    if len(words) == 1 and words[0] in QUALIFIERS:
        return ""
    return SYNONYMS.get(phrase, phrase)


_MEASURE_KIND = frozenset({"sopa", "cha", "cafe", "sobremesa"})
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_WORD_NUMBERS = {"um": 1.0, "uma": 1.0, "dois": 2.0, "duas": 2.0, "tres": 3.0, "meia": 0.5, "meio": 0.5}
_QTY = re.compile(
    r"^(?:(?P<mixed_int>\d+)\s*(?:e\s+)?(?P<mixed_num>\d+)/(?P<mixed_den>\d+)"
    r"|(?P<num>\d+)/(?P<den>\d+)"
    r"|(?P<int_frac>\d+)?\s*(?P<uni>[½¼¾⅓⅔])"
    r"|(?P<dec>\d+(?:[.,]\d+)?))\s*"
)
_CUTS = re.compile(
    r"(\s+a\s+gosto\b|\s+at[eé]\s+|\s+para\s+|\s+em\s+(?!p[oó]\b)|,|\s+com\b|\s+ou\s+|\s+--|;|\s+q\.?b\.?)",
    re.IGNORECASE,
)


def _parse_quantity(text: str) -> tuple[float | None, str]:
    m = _QTY.match(text)
    if m and m.group(0).strip():
        g = m.groupdict()
        if g["mixed_int"] is not None:
            q = int(g["mixed_int"]) + int(g["mixed_num"]) / max(int(g["mixed_den"]), 1)
        elif g["num"] is not None:
            q = int(g["num"]) / max(int(g["den"]), 1)
        elif g["uni"] is not None:
            q = (int(g["int_frac"]) if g["int_frac"] else 0) + _FRACTIONS[g["uni"]]
        else:
            q = float(g["dec"].replace(",", "."))
        return q, text[m.end():]
    first = strip_accents(text.split(" ", 1)[0].lower())
    if first in _WORD_NUMBERS and " " in text:
        return _WORD_NUMBERS[first], text.split(" ", 1)[1]
    return None, text


def _is_prose(text: str) -> bool:
    plain = strip_accents(text.lower())
    if re.search(r"\b[a-z]+-se\b", plain):
        return True
    first = re.findall(r"[a-z]+", plain)
    return bool(first) and first[0] in PREP_VERBS


def parse_ingredient_line(line: str) -> list[ParsedIngredient]:
    """Converte uma linha de ingrediente em 0..n itens. Rótulos e prosa devolvem []."""
    raw = line
    text = html.unescape(line).replace("\xa0", " ")
    text = re.sub(r"^[\s*#:]+", "", text).strip()
    if not text or _is_prose(text):
        return []
    text = re.sub(r"\([^)]*\)", " ", text).replace("(", " ").replace(")", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*(?:(?:bem|mal)\s+)?(?:cheias?|cheios?|rasas?|rasos?)\s*,?\s*", " ", text,
                  flags=re.IGNORECASE)
    is_label = text.endswith(":")
    text = re.sub(r"(--|[;.,:\s])+$", "", text).strip()

    quantity, rest = _parse_quantity(text)
    if is_label and quantity is None:
        return []
    rest = rest.strip()
    unit = None
    if quantity is not None:
        words = rest.split(" ")
        if words and strip_accents(words[0].lower()).strip(".,;") in UNITS:
            unit = UNITS[strip_accents(words[0].lower()).strip(".,;")]
            words = words[1:]
            plain = [strip_accents(w.lower()) for w in words]
            if len(plain) > 1 and plain[0] == "de" and plain[1] in _MEASURE_KIND:
                words = words[2:]
            elif plain and plain[0] in _MEASURE_KIND:
                words = words[1:]
            while words and strip_accents(words[0].lower()) in UNIT_SIZES:
                words = words[1:]
        rest = " ".join(words)
    rest = re.sub(r"^(de|do|da|dos|das)\b\s*", "", rest.strip(), flags=re.IGNORECASE)

    if quantity is None and re.search(r",.*\s+e\s+", rest):
        # lista sem quantidade: "Sal, pimenta e queijo ralado a gosto"
        rest = re.sub(r"\s*,\s*", " e ", rest)
    rest = _CUTS.split(rest, maxsplit=1)[0].strip()
    parts = re.split(r"\s+e\s+", rest)
    result: list[ParsedIngredient] = []
    for i, part in enumerate(parts):
        name = re.sub(r"^(o|a|os|as|um|uma)\s+", "", part.strip(" .,;:-"), flags=re.IGNORECASE)
        key = ingredient_key(name)
        if not key or len(key.split()) > 6:
            continue
        q, u = (quantity, unit) if i == 0 else (None, None)
        result.append(ParsedIngredient(raw=raw, name=name, key=key, quantity=q, unit=u))
    return result
