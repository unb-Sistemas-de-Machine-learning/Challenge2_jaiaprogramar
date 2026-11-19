"""Coleta (API MediaWiki) e leitura do wikitexto do "Livro de receitas" do Wikilivros.

Licença do conteúdo: CC BY-SA 4.0. A coleta respeita o limite da Wikimedia para
clientes sem identificação de contato (10 requisições/min) e o `Retry-After`.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ingredients import strip_accents

API = "https://pt.wikibooks.org/w/api.php"
USER_AGENT = "ZeroDesperdicio-academico/0.1 (projeto universitario de IA; leitura via API)"
PREFIX = "Livro de receitas/"
BATCH = 50


class RateLimited(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"limite de requisições; tentar de novo em {retry_after}s")
        self.retry_after = retry_after


@dataclass
class RawRecipe:
    variant: int
    label: str
    ingredient_lines: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ markup

def clean_markup(text: str) -> str:
    t = html.unescape(text).replace("\xa0", " ")
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\{\{\s*w\s*\|([^{}|]*)\|([^{}]*)\}\}", r"\2", t)
    t = re.sub(r"\{\{\s*w\s*\|([^{}]*)\}\}", r"\1", t)
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:Imagem|Image|Ficheiro|File|Arquivo|Categoria):[^\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"'{2,}", "", t)
    return re.sub(r"\s+", " ", t).strip()


def page_url(title: str) -> str:
    return "https://pt.wikibooks.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="/:")


# ------------------------------------------------------------------ parser

_IGNORE_PREFIXES = ("dica", "nota", "observ", "rendimento", "ligac", "referenc", "ver tambem",
                    "esta lista", "fonte", "curiosidade", "historia", "obs")


def _classify(label: str) -> str:
    s = strip_accents(label.lower()).strip(" :.")
    if s.startswith(_IGNORE_PREFIXES):
        return "ignore"
    if "ingrediente" in s:
        return "ingredients"
    if "prepar" in s or "modo de fazer" in s or "como fazer" in s or "montagem" in s:
        return "steps"
    return "other"


def _categories(wikitext: str) -> list[str]:
    cats: list[str] = []
    best: list[str] = []
    for nav in re.findall(r"\{\{Navegação\|(.*?)\}\}", wikitext, re.S):
        for path in re.findall(r"Livro de receitas/Índice/([^|\]]+)", nav) + re.findall(r"\[\[\.\./([^|\]]+)", nav):
            parts = [p.strip() for p in path.split("/") if p.strip()]
            if len(parts) > len(best):
                best = parts
    cats.extend(best)
    for c in re.findall(r"\[\[Categoria:([^|\]]+)", wikitext):
        c = c.strip()
        if ", receitas com" in c or "{{" in c or not c:
            continue
        cats.append(c)
    seen, out = set(), []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")
_LABEL = re.compile(r"^[:;]*\s*'''(.+?)'''\s*(.*)$")
_ONLY_LINK = re.compile(r"^\[\[Livro de receitas/[^\]]+\]\]\s*[.;,]?$")


def parse_page(wikitext: str) -> tuple[list[RawRecipe], list[str]]:
    """Separa as variações de receita de uma página. Devolve (receitas, categorias)."""
    blocks: list[RawRecipe] = [RawRecipe(variant=0, label="")]
    mode = "ingredients"
    for raw_line in wikitext.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("{|", "|", "!", "[[Categoria", "[[Imagem", "[[Image", "[[Ficheiro")):
            continue
        if re.fullmatch(r"(\{\{[^{}]*\}\}\s*)+", line):
            continue
        m = _HEADING.match(line)
        if m:
            level, text = len(m.group(1)), clean_markup(m.group(2))
            kind = _classify(text)
            if level <= 2 and kind == "other":
                cur = blocks[-1]
                if cur.ingredient_lines or cur.steps:
                    blocks.append(RawRecipe(variant=0, label=text))
                else:
                    cur.label = text
                mode = "ingredients"
            else:
                mode = "ingredients" if kind == "other" else kind
            continue
        lm = _LABEL.match(line)
        if lm and (lm.group(1).strip().endswith(":") or not lm.group(2).strip() or lm.group(2).lstrip().startswith(":")):
            kind = _classify(clean_markup(lm.group(1)))
            mode = "ingredients" if kind == "other" else kind
            trailing = clean_markup(lm.group(2)).lstrip(": ")
            if trailing and mode == "steps":
                blocks[-1].steps.append(trailing)
            continue
        cur = blocks[-1]
        if mode == "ignore":
            continue
        if line.startswith("#"):
            text = clean_markup(line.lstrip("#*: "))
            if text:
                cur.steps.append(text)
        elif line.startswith("*"):
            body = line.lstrip("* ").strip()
            if _ONLY_LINK.match(body):
                continue
            text = clean_markup(body)
            if not text:
                continue
            (cur.ingredient_lines if mode == "ingredients" else cur.steps).append(text)
        elif mode == "steps":
            text = clean_markup(line.lstrip(":; "))
            if text:
                cur.steps.append(text)
    recipes = [b for b in blocks if b.ingredient_lines]
    for i, r in enumerate(recipes, start=1):
        r.variant = i
    return recipes, _categories(wikitext)


# ------------------------------------------------------------------ coleta

def _http_request(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(API, data=data, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            raise RateLimited(int(e.headers.get("Retry-After", "10"))) from e
        raise


def _with_retry(request, params, sleep, log, tries: int = 8) -> dict:
    for _ in range(tries):
        try:
            return request(params)
        except RateLimited as e:
            log(f"  limite atingido; aguardando {e.retry_after + 2}s")
            sleep(e.retry_after + 2)
    raise RuntimeError("a API recusou o pedido várias vezes seguidas; tente mais tarde")


def fetch_all(out_path: Path, *, request: Callable[[dict], dict] = _http_request,
              sleep: Callable[[float], None] = time.sleep, pace: float = 6.5,
              log: Callable[..., None] = print) -> int:
    """Baixa todas as páginas do livro (retomável). Devolve o total de páginas no arquivo."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    titles: list[str] = []
    cont: dict = {}
    while True:
        d = _with_retry(request, {"action": "query", "list": "allpages", "apprefix": PREFIX,
                                  "aplimit": "500", "apfilterredir": "nonredirects", **cont}, sleep, log)
        titles += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            break
        cont = d["continue"]
        sleep(pace)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(l)["title"] for l in f if l.strip()}
    todo = [t for t in titles if t not in done]
    log(f"títulos: {len(titles)} | já salvos: {len(done)} | faltam: {len(todo)}")
    with out_path.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH):
            d = _with_retry(request, {"action": "query", "prop": "revisions", "rvprop": "content|ids|timestamp",
                                      "rvslots": "main", "titles": "|".join(todo[i:i + BATCH])}, sleep, log)
            for p in d["query"]["pages"]:
                if "revisions" not in p:
                    continue
                rev = p["revisions"][0]
                out.write(json.dumps({"title": p["title"], "pageid": p["pageid"], "revid": rev["revid"],
                                      "timestamp": rev["timestamp"],
                                      "wikitext": rev["slots"]["main"]["content"]}, ensure_ascii=False) + "\n")
            out.flush()
            log(f"  lote {i // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}")
            sleep(pace)
    with out_path.open(encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())
