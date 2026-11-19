# ZeroDesperdício — MVP de demonstração (RAG + agentes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ter amanhã (11/11/2026) uma aplicação rodando que recebe a despensa da pessoa, prioriza o que vence e sugere receitas reais com justificativa e fonte, para mostrar ao professor.

**Architecture:** Pipeline de dados em camadas (raw → processed → index) a partir do Wikilivros; recuperação híbrida (semântica com embeddings + lexical por ingredientes) seguida de ranking determinístico; agentes simples (Despensa, Restrições, Receitas) coordenados por um orquestrador que registra o rastro; LLM opcional só para redigir a explicação. Interface em Streamlit.

**Tech Stack:** Python 3.10, uv, Streamlit 1.65, ChromaDB 1.5, fastembed 0.9 (ONNX, CPU), SDK `anthropic` 2.x, pytest.

**Spec:** `ZeroDesperdicio_visao_geral.pdf` (visão geral aprovada pelo usuário) + decisões desta conversa, resumidas abaixo.

## Global Constraints

- Demonstração em 2026-11-11; entrega final do desafio em 2026-12-07.
- Máquina alvo: WSL2, Python 3.10.12, 8 núcleos, ~5 GB de RAM, sem GPU. Nada de PyTorch nesta etapa.
- **Regra de ouro:** validade, prioridade, restrições e ranking são código determinístico. O LLM só redige a explicação e nunca afirma que um alimento está seguro.
- Item vencido nunca entra em sugestão de receita.
- Toda sugestão mostra a fonte: URL da receita, data da revisão e licença.
- Dados só de fonte com licença clara: Wikilivros "Livro de receitas", CC BY-SA 4.0, pela API oficial (`https://pt.wikibooks.org/w/api.php`). Limite sem contato no User-Agent: 10 requisições/min → 1 a cada 6,5 s; em 429/503, esperar o `Retry-After`. O e-mail do usuário não vai no User-Agent.
- Receitas da categoria "Bebidas com álcool" ficam fora da base (decisão padrão, registrada no manifesto).
- O app funciona sem credencial de LLM. Falha do LLM nunca derruba a tela: cai na explicação por regras e mostra o motivo.
- LLM padrão: `claude-opus-5-5` pelo SDK oficial `anthropic`, `output_config={"effort": "low"}`, `max_tokens=16000`, `fallbacks="default"` com beta `server-side-fallback-2026-07-01`; checar `stop_reason == "refusal"` antes de ler o conteúdo. Modelo trocável por `ZD_LLM_MODEL`.
- Interface e documentação em pt-BR; identificadores de código em inglês.
- Testes unitários não usam rede nem baixam modelo.
- Sem `git init` e sem commits nesta rodada (decisão do usuário pendente). Cada tarefa termina com a suíte verde.
- Fora do escopo desta rodada: agente de Visão (foto → alimentos, leitura da validade, voz — semana 4), MLflow, DVC, CI, Docker, monitoramento e retreino (semana 5), orquestração por LLM com tool use.

## Review Focus

Condições que a visão geral implica e que mais podem aparecer na demonstração. Cada uma tem teste na tarefa dona do código:

1. Despensa vazia, ou só com itens vencidos → orientação na tela, sem exceção (Tarefa 7).
2. Base ou índice ausentes (colega abre o app antes de rodar o pipeline) → tela com os comandos para gerar, sem stack trace (Tarefa 9).
3. LLM falha: sem rede, 401, 429, recusa, ou cita receita que não existe → explicação por regras + aviso (Tarefa 6).
4. Entrada estranha: nome vazio, só números, emoji, plural, acento, item que não existe na base → normaliza ou marca "não reconhecido" (Tarefas 1 e 7).
5. Arquivo da despensa corrompido → começa vazio e avisa (Tarefa 2).

---

## Estrutura de arquivos

```
pyproject.toml  Makefile  .gitignore  .env.example  README.md  .streamlit/config.toml
src/zerodesperdicio/
  config.py            caminhos e constantes
  ingredient_rules.py  tabelas: unidades, qualificadores, compostos protegidos, sinônimos, básicos
  ingredients.py       linha de ingrediente → nome, chave, quantidade, unidade
  pantry.py            itens, status de validade, urgência, persistência em JSON
  recipes.py           modelo Recipe, leitura/gravação JSONL, texto para embedding
  wikibooks.py         cliente da API (retomável) + parser de wikitexto
  pipeline.py          raw → processed + manifesto (CLI)
  ranking.py           índice lexical, restrições, pontuação
  index.py             embeddings + ChromaDB (CLI)
  llm.py               cliente Claude + detecção de credencial
  explain.py           contexto, explicação por regras, explicação com LLM
  agents.py            agentes, orquestrador, rastro, fábrica
  evaluation.py        recall@k / MRR e comparação LLM × RAG (CLI)
app/streamlit_app.py   interface
tests/                 um arquivo de teste por módulo
docs/adr/              decisões de arquitetura (status "Proposto")
data/{raw/wikibooks,processed,index,user}/   gerados, fora do versionamento
reports/retrieval_eval.json                   métricas da recuperação
```

---

### Task 1: Projeto e normalização de ingredientes

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `.env.example`, `.streamlit/config.toml`
- Create: `src/zerodesperdicio/__init__.py`, `config.py`, `ingredient_rules.py`, `ingredients.py`
- Test: `tests/test_ingredients.py`

**Interfaces:**
- Produces:
  - `config`: `ROOT`, `RAW_PAGES`, `RECIPES`, `MANIFEST`, `INDEX_DIR`, `INDEX_MANIFEST`, `PANTRY_FILE`, `MODEL_CACHE`, `REPORTS_DIR` (todos `Path`; `ROOT` vem de `ZD_HOME` ou da raiz do repositório), `EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`.
  - `ingredient_rules`: `UNITS: dict[str, str]` (forma sem acento → unidade canônica), `QUALIFIERS: frozenset[str]`, `PROTECTED: frozenset[str]`, `SYNONYMS: dict[str, str]`, `STAPLES: frozenset[str]` = `{"agua", "sal", "acucar", "oleo", "pimenta do reino"}`.
  - `ingredients.ParsedIngredient(raw: str, name: str, key: str, quantity: float | None, unit: str | None)` (frozen dataclass).
  - `ingredients.strip_accents(text: str) -> str`
  - `ingredients.ingredient_key(name: str) -> str` — minúsculas, sem acento, singular, sem qualificadores (nunca remove a primeira palavra), sinônimos aplicados; `""` se não sobrar nada alfabético.
  - `ingredients.parse_ingredient_line(line: str) -> list[ParsedIngredient]` — 0..n itens; divide "sal e pimenta"; devolve `[]` para rótulo, linha vazia ou instrução em prosa.

- [ ] **Step 1: Ambiente** — `pyproject.toml` (setuptools, layout `src/`, dependências `streamlit`, `chromadb`, `fastembed`, `anthropic`, `python-dotenv`, `numpy`, `pandas`; extra `dev` com `pytest`), depois `uv sync --extra dev`. `.streamlit/config.toml` com `server.headless = true` e `browser.gatherUsageStats = false`.
  Verificar: `.venv/bin/python -c "import zerodesperdicio"` sai com código 0.

- [ ] **Step 2: Teste que falha** — `tests/test_ingredients.py`, parametrizado. Cada caso compara `[(p.key, p.quantity, p.unit) for p in parse_ingredient_line(linha)]`:

```python
CASES = [
    ("3 xícaras (chá) de açúcar", [("acucar", 3.0, "xicara")]),
    ("1 e 1/2 copos (americano) de leite", [("leite", 1.5, "copo")]),
    ("2 colheres (sopa) de manteiga derretida", [("manteiga", 2.0, "colher")]),
    ("4 ovos;", [("ovo", 4.0, None)]),
    ("Sal a gosto", [("sal", None, None)]),
    ("Óleo para fritar.", [("oleo", None, None)]),
    ("1 kg de bacalhau, dessalgado --", [("bacalhau", 1.0, "kg")]),
    ("1 pitada de amido de milho (maisena)", [("amido de milho", 1.0, "pitada")]),
    ("1 lata de leite condensado", [("leite condensado", 1.0, "lata")]),
    ("3 claras", [("ovo", 3.0, None)]),
    ("Sal e pimenta do reino a gosto", [("sal", None, None), ("pimenta do reino", None, None)]),
    ("2 tomates maduros picados", [("tomate", 2.0, None)]),
    ("500 gramas de carne moída;", [("carne moida", 500.0, "g")]),
    ("1 pacote pequeno de maisena (200 gramas)", [("amido de milho", 1.0, "pacote")]),
    ("1 colher (sopa) de fermento em pó (Pó Royal);", [("fermento em po", 1.0, "colher")]),
    ("2 tomates cortados em cubos", [("tomate", 2.0, None)]),
    ("1 molho de coentro", [("coentro", 1.0, "maco")]),
    ("½ xícara de leite", [("leite", 0.5, "xicara")]),
    ("1 lata de creme de leite (sem o soro);", [("creme de leite", 1.0, "lata")]),
    ("250 ml de leite de coco", [("leite de coco", 250.0, "ml")]),
    ("1&nbsp;kg de farinha de trigo", [("farinha de trigo", 1.0, "kg")]),
]
NOT_INGREDIENTS = ["", "*", "Para a massa:", "2 colheres (sopa) de ;",
    "Mexe-se bem, junta-se 1 colher de manteiga e algum leite, se estiver muito duro."]
KEYS = [("Tomates", "tomate"), ("iogurte natural", "iogurte"), ("Ovos", "ovo"),
    ("muçarela", "queijo mussarela"), ("pães", "pao"), ("limões", "limao"),
    ("leite integral", "leite"), ("  ", ""), ("123", ""), ("🍅", "")]
```
  Também: `parse_ingredient_line("3 xícaras (chá) de açúcar")[0].name == "açúcar"`.

- [ ] **Step 3:** `.venv/bin/python -m pytest tests/test_ingredients.py -q` → falha por módulo inexistente.
- [ ] **Step 4: Implementar** `ingredient_rules.py` e `ingredients.py`. Ordem dentro de `parse_ingredient_line`: desfazer entidades HTML → tirar parênteses e pontuação final → extrair quantidade (inteiro, decimal com vírgula, fração, "1 e 1/2", ½ ¼ ¾, "meia") → unidade com adjetivo de tamanho opcional → "de" → cortar em vírgula, " para ", " a gosto" e " em " (exceto "em pó") → dividir em " e " → chave. Linha é prosa se tiver verbo com "-se", começar por verbo de preparo ou a chave passar de 6 palavras.
- [ ] **Step 5:** mesma linha de pytest → tudo passa.

### Task 2: Despensa

**Files:**
- Create: `src/zerodesperdicio/pantry.py`
- Test: `tests/test_pantry.py`

**Interfaces:**
- Consumes: `ingredients.ingredient_key`.
- Produces:
  - `PantryItem(name: str, quantity: str = "", expires_on: date | None = None)`
  - `ExpiryStatus` (str Enum): `EXPIRED="vencido"`, `TODAY="vence hoje"`, `SOON="vence em breve"`, `OK="dentro da validade"`, `UNKNOWN="sem data"`
  - `PantryEntry(item, key: str, days_left: int | None, status: ExpiryStatus, urgency: float)` (frozen)
  - `SOON_DAYS = 3`, `URGENCY_HORIZON_DAYS = 14`
  - `urgency(days_left: int | None) -> float` — `None` e negativos → 0.0; senão `max(0, 1 - d/14)`
  - `assess(items: list[PantryItem], today: date) -> list[PantryEntry]` — ordenado por `days_left` crescente, sem data por último
  - `split_expired(entries) -> tuple[list[PantryEntry], list[PantryEntry]]` — (utilizáveis, vencidos)
  - `PantryState(items: list[PantryItem], used: int = 0, discarded: int = 0)`
  - `PantryStore(path: Path)`: `load() -> tuple[PantryState, str | None]` (estado, aviso), `save(state) -> None`

- [ ] **Step 1: Testes que falham**, com `today = date(2026, 11, 10)`:
  - validade 09/11 → `EXPIRED`, `days_left == -1`, `urgency == 0.0`
  - 10/11 → `TODAY`, `urgency == 1.0`; 13/11 → `SOON`; 14/11 → `OK`; sem data → `UNKNOWN`, `urgency == 0.0`
  - `urgency(7) == 0.5`, `urgency(14) == 0.0`, `urgency(30) == 0.0`
  - `assess` ordena `[sem data, 13/11, 09/11, 10/11]` como `[09/11, 10/11, 13/11, sem data]`
  - `split_expired` separa o item de 09/11
  - `PantryStore`: salvar e carregar devolve o mesmo estado (datas incluídas); arquivo inexistente → estado vazio e aviso `None`; arquivo com `"{isso não é json"` → estado vazio e aviso não vazio (Review Focus 5).
- [ ] **Step 2:** `pytest tests/test_pantry.py -q` → falha.
- [ ] **Step 3: Implementar** `pantry.py` (JSON com datas ISO; gravação atômica por arquivo temporário + `replace`).
- [ ] **Step 4:** pytest → passa.

### Task 3: Ingestão do Wikilivros e pipeline de dados

**Files:**
- Create: `src/zerodesperdicio/recipes.py`, `wikibooks.py`, `pipeline.py`
- Test: `tests/test_wikibooks.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `parse_ingredient_line`, `config`.
- Produces:
  - `recipes.Recipe(id, title, ingredients: list[str], ingredient_names: list[str], ingredient_keys: list[str], steps: list[str], categories: list[str], url, revision_id: int, revision_date: str, source: str, license: str)` (frozen)
  - `recipes.load_recipes(path) -> list[Recipe]`, `recipes.save_recipes(recipes, path) -> None`, `recipes.embedding_text(recipe) -> str` (`"Título. Categoria: a, b. Ingredientes: x, y"`)
  - `wikibooks.RawRecipe(variant: int, label: str, ingredient_lines: list[str], steps: list[str])`
  - `wikibooks.clean_markup(text: str) -> str` — `[[a|b]]`→`b`, `{{w|x}}`→`x`, demais predefinições e `<ref>` removidos, negrito/itálico removidos, entidades HTML desfeitas
  - `wikibooks.parse_page(wikitext: str) -> tuple[list[RawRecipe], list[str]]` — (receitas, categorias)
  - `wikibooks.page_url(title: str) -> str`
  - `wikibooks.fetch_all(out_path: Path, *, request=..., sleep=time.sleep, pace: float = 6.5, log=print) -> int` — lista títulos com prefixo `Livro de receitas/`, baixa em lotes de 50, pula títulos já gravados, espera `Retry-After` em 429/503; devolve o total no arquivo. `request(params: dict) -> dict` é injetável; levanta `wikibooks.RateLimited(retry_after: int)`.
  - `pipeline.build_recipes(pages: Iterable[dict]) -> tuple[list[Recipe], dict]` — estatísticas com as chaves `pages`, `pages_without_recipe`, `parsed`, `dropped_alcohol`, `dropped_few_ingredients`, `dropped_no_steps`, `dropped_duplicates`, `recipes`, `ingredient_lines`, `lines_without_key`, `unique_keys`, `top_keys`
  - `pipeline.run_build() -> dict` — grava `RECIPES` e `MANIFEST` (fonte, licença, sha256 do raw e do processado, sha256 de `ingredient_rules.py`, estatísticas, data)
  - CLI: `python -m zerodesperdicio.pipeline fetch|build`

Regras do parser (máquina de estados por linha; modo inicial de cada bloco = ingredientes):
- Título de nível 2 que não seja palavra-chave abre novo bloco (variante).
- Qualquer título ou rótulo em negrito: contém "ingrediente" → ingredientes; contém "prepar", "modo de fazer", "como fazer" ou "montagem" → passos; começa com dica/nota/observ/rendimento/ligações/referências/ver também/"esta lista" → ignorar; outro rótulo ("Para a massa", "Recheio") → ingredientes.
- `#` é sempre passo. `*` é ingrediente no modo ingredientes e passo no modo passos. Parágrafo só conta no modo passos.
- Bullet que é só link para outra página do livro é descartado.
- Receita precisa de ≥ 2 chaves de ingrediente distintas e ≥ 1 passo.

- [ ] **Step 1: Testes que falham** em `tests/test_wikibooks.py`, com wikitexto de exemplo no próprio teste:
  - Layout A com duas variantes (`== Churros - 1 ==` / `== Churros - 2 ==`, `==='''Ingredientes e Preparo:'''===`, bullets terminando em `--` e `;`, `:'''Preparo:'''`, passos `#`): 2 receitas, a primeira com 3 linhas de ingrediente e 2 passos; categorias `["Doces", "Outros Doces"]` vindas da `{{Navegação|...}}`.
  - Layout B (`== Receita 1 ==`, `=== Ingredientes ===`, `=== Preparo ===`, `=== Ingredientes do recheio ===`, `=== Preparo do recheio ===` com passo em parágrafo `: Faça um refogado...`, `=== Montagem ===`): 1 receita com os ingredientes dos dois grupos e os passos na ordem.
  - `* 1 {{w|coco}} ralado;` vira linha `1 coco ralado`.
  - Rótulo `'''Rendimento:''' Em torno de 100 unidades.` não entra em ingredientes nem em passos.
  - Página de categoria (só `== Esta lista é gerada automaticamente ==` e bullets de links) → nenhuma receita.
  - Categorias: `[[Categoria:Bolos|{{SUBPAGENAME}}]]` entra; `[[Categoria:Leite, receitas com|...]]` não.
  - `fetch_all` com `request` falso: (a) grava todas as páginas de 120 títulos em 3 lotes; (b) segunda chamada não pede nenhum título já gravado; (c) `RateLimited(36)` na primeira tentativa → `sleep` chamado com valor ≥ 36 e o lote é repetido.
- [ ] **Step 2: Testes que falham** em `tests/test_pipeline.py`: `build_recipes` com 4 páginas sintéticas (uma válida, uma duplicata exata da válida em outra página, uma com categoria "Bebidas com álcool", uma de categoria sem receita) → `stats["recipes"] == 1`, `dropped_duplicates == 1`, `dropped_alcohol == 1`, `pages_without_recipe == 1`; a receita tem `id` no formato `wb-<pageid>-<variante>`, `url` começando por `https://pt.wikibooks.org/wiki/Livro_de_receitas/`, `license == "CC BY-SA 4.0"` e `ingredient_keys` sem repetição. `save_recipes` → `load_recipes` devolve objetos iguais.
- [ ] **Step 3:** pytest dos dois arquivos → falha.
- [ ] **Step 4: Implementar** os três módulos.
- [ ] **Step 5:** pytest → passa.
- [ ] **Step 6: Rodar de verdade** — `.venv/bin/python -I -m zerodesperdicio.pipeline fetch` (≈ 5 min, em segundo plano) e depois `build`. Esperado: 2058 páginas no raw; manifesto com mais de 1.500 receitas.
- [ ] **Step 7: Qualidade da normalização** — olhar as 150 chaves mais frequentes e as linhas sem chave; cada regra nova entra primeiro como caso em `tests/test_ingredients.py`. Depois, inspeção manual de 100 linhas aleatórias (semente fixa) e registro da taxa de acerto no manifesto (`manual_sample_accuracy`).

### Task 4: Ranking determinístico

**Files:**
- Create: `src/zerodesperdicio/ranking.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Consumes: `Recipe`, `PantryEntry`, `ingredient_key`, `STAPLES`.
- Produces:
  - `Weights(coverage: float = 0.5, urgency: float = 0.4, semantic: float = 0.1)`; `Weights.for_preference()` → `(0.4, 0.3, 0.3)`
  - `Suggestion(recipe, score, coverage, urgency_share, semantic, used: list[PantryEntry], missing: list[str], staples: list[str], reasons: list[str])`
  - `LexicalIndex(recipes)`: `search(keys: set[str], k: int) -> list[tuple[str, int]]` (id, nº de chaves em comum; decrescente, empate por id); `known_keys: frozenset[str]`
  - `violated_restriction(recipe, restrictions: list[str]) -> str | None` — compara por palavras: a restrição "leite" barra "leite condensado" e "creme de leite" (errar para o lado seguro)
  - `score_recipe(recipe, pantry: list[PantryEntry], *, staples: frozenset[str], semantic: float, weights: Weights) -> Suggestion | None` — `None` se a receita não usa nenhum item da despensa
  - `rank(candidates: list[tuple[Recipe, float]], pantry, *, restrictions, staples, weights, top_n) -> tuple[list[Suggestion], int]` — (sugestões, removidas por restrição)

Fórmulas: `coverage = |chaves ∩ (despensa ∪ básicos)| / |chaves|`; `urgency_share = Σ urgência dos itens usados / Σ urgência de toda a despensa` (0 se o denominador for 0); `score = w.coverage·coverage + w.urgency·urgency_share + w.semantic·max(0, semantic)`. Desempate: menos ingredientes faltando, depois id.

- [ ] **Step 1: Testes que falham**:
  - despensa {ovo (vence hoje), tomate (7 dias), queijo mussarela (sem data)}, receita {ovo, tomate, sal, cebola}, básicos padrão, `semantic=0.5`, pesos padrão → `coverage == 0.75`, `urgency_share == 1.0`, `score == pytest.approx(0.5*0.75 + 0.4*1.0 + 0.1*0.5)`, `missing == ["cebola"]`, `staples == ["sal"]`, `used` na ordem de urgência (ovo, tomate)
  - receita só com {sal, agua} → `None`
  - `staples=frozenset()` → `coverage == 0.5`
  - despensa sem nenhuma data → `urgency_share == 0.0`
  - `reasons` da primeira sugestão contém "vence hoje" e "3 de 4"
  - `violated_restriction`: "leite" barra receita com "leite condensado"; "amendoim" não barra receita sem amendoim; restrição `"  "` é ignorada
  - `rank`: a receita que usa o item que vence hoje fica acima da que só usa o item sem data; receita com ingrediente restrito sai e o contador devolve 1; `top_n` respeitado
  - `LexicalIndex.search({"ovo", "tomate"}, 2)` devolve primeiro a receita com as duas chaves
- [ ] **Step 2:** pytest → falha. **Step 3:** implementar. **Step 4:** pytest → passa.

### Task 5: Índice vetorial

**Files:**
- Create: `src/zerodesperdicio/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `Recipe`, `embedding_text`, `config`.
- Produces:
  - `Embedder` (Protocol): `name: str`; `embed(texts: list[str]) -> np.ndarray` (n×d, float32, norma 1)
  - `FastEmbedder(model_name=config.EMBEDDING_MODEL, cache_dir=config.MODEL_CACHE)`
  - `VectorIndex(path: Path, embedder: Embedder)`: `build(recipes, recipes_sha256: str) -> dict`, `search(query: str, k: int) -> list[tuple[str, float]]` (id, similaridade de cosseno), `similarities(query: str, ids: list[str]) -> dict[str, float]`, `count() -> int`
  - `index.read_manifest(path) -> dict | None`
  - CLI: `python -m zerodesperdicio.index build`

Chroma: `PersistentClient` com `anonymized_telemetry=False`; coleção `recipes` em espaço de cosseno, `embedding_function=None` (os vetores são sempre passados prontos); `build` apaga e recria a coleção e grava `index_manifest.json` (modelo, dimensão, quantidade, `recipes_sha256`, data).

- [ ] **Step 1: Testes que falham** com `FakeEmbedder` (bag-of-words por hash, 64 dimensões, normalizado) e `tmp_path`:
  - `build` com 4 receitas → `count() == 4`; segunda `build` continua em 4
  - `search("bolo de cenoura", 2)[0][0]` é o id da receita "Bolo de cenoura", com similaridade entre 0 e 1
  - `similarities(query, [id1, id2])` devolve as duas chaves, e o valor de `id1` bate com o de `search` (`pytest.approx`, `abs=1e-4`)
  - índice vazio: `search` devolve `[]`
  - `read_manifest` devolve o `recipes_sha256` passado; pasta sem manifesto → `None`
- [ ] **Step 2:** pytest → falha. **Step 3:** implementar. **Step 4:** pytest → passa.
- [ ] **Step 5: Rodar de verdade** — `.venv/bin/python -I -m zerodesperdicio.index build` (baixa o modelo de ~220 MB na primeira vez). Esperado: quantidade igual à do manifesto dos dados; uma busca por "bolo de cenoura" devolve um bolo.

### Task 6: LLM e explicação

**Files:**
- Create: `src/zerodesperdicio/llm.py`, `explain.py`
- Test: `tests/test_llm.py`, `tests/test_explain.py`

**Interfaces:**
- Consumes: `Suggestion`, `PantryEntry`.
- Produces:
  - `llm.LLMError(Exception)` — mensagem em português pronta para a tela
  - `llm.LLMClient` (Protocol): `name: str`; `complete(system: str, user: str) -> str`
  - `llm.ClaudeLLM(client: anthropic.Anthropic | None = None, model: str | None = None)`
  - `llm.FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})` — só para esses o pedido vai por `client.beta.messages.create` com `betas=["server-side-fallback-2026-07-01"]` e `fallbacks="default"`; os demais usam `client.messages.create`
  - `llm.get_llm() -> LLMClient | None` — carrega `.env`; `None` quando o cliente não tem `api_key`, `auth_token` nem `credentials`
  - `explain.Explanation(text: str, source: Literal["llm", "regras"], error: str | None = None)`
  - `explain.build_context(suggestions, entries) -> str` — receitas como `[R1]`, `[R2]`… com título, itens usados (e dias até vencer), faltantes, URL
  - `explain.rule_based(suggestions, entries) -> str`
  - `explain.explain(suggestions, entries, llm: LLMClient | None) -> Explanation`

- [ ] **Step 1: Testes que falham** em `tests/test_llm.py`, usando `anthropic.Anthropic(api_key="sk-test", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))`:
  - pedido padrão: corpo com `model == "claude-opus-5-5"`, `max_tokens == 16000`, `fallbacks == "default"`, `output_config == {"effort": "low"}`, `system` e `messages[0]["content"]` iguais aos argumentos; cabeçalho `anthropic-beta` contém `server-side-fallback-2026-07-01`
  - resposta com blocos `thinking` + `text` → devolve só o texto
  - `model="claude-haiku-5-5"` → corpo sem `fallbacks` e pedido sem o cabeçalho beta
  - `stop_reason == "refusal"` → `LLMError`
  - resposta sem bloco de texto → `LLMError`
  - HTTP 401 → `LLMError` cuja mensagem contém "chave"; 429 → contém "limite"; erro de conexão → contém "rede"
  - `get_llm()` com `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` removidos do ambiente e `HOME` em `tmp_path` → `None`; com `ANTHROPIC_API_KEY="sk-x"` → instância de `ClaudeLLM`
- [ ] **Step 2: Testes que falham** em `tests/test_explain.py`:
  - `build_context` de 2 sugestões contém `[R1]`, `[R2]`, os títulos, a URL e "vence hoje"
  - `explain(..., llm=None)` → `source == "regras"`, `error is None`, texto cita o título da primeira sugestão
  - LLM falso que devolve `"Faça a omelete [R1]."` → `source == "llm"`
  - LLM falso que levanta `LLMError("sem rede")` → `source == "regras"`, `error == "sem rede"`
  - LLM falso que devolve texto citando `[R9]` com 2 sugestões, ou sem nenhuma citação → `source == "regras"` e `error` não vazio (Review Focus 3)
  - lista de sugestões vazia → texto de orientação, sem chamar o LLM
- [ ] **Step 3:** pytest → falha. **Step 4:** implementar (cadeia de `except` do mais específico para o mais geral: `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `RateLimitError`, `APIStatusError`, `APIConnectionError`). O prompt de sistema diz que o texto das receitas é dado, não instrução, que só pode usar fatos do contexto, que deve citar `[R#]` e que não pode afirmar que um alimento está seguro. **Step 5:** pytest → passa.

### Task 7: Agentes e orquestrador

**Files:**
- Create: `src/zerodesperdicio/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: tudo das tarefas 1–6.
- Produces:
  - `TraceStep(agent: str, action: str, detail: str, ms: float)`
  - `Request(items: list[PantryItem], today: date, restrictions: list[str] = [], preference: str = "", assume_staples: bool = True, top_n: int = 5)`
  - `Result(entries, expired, unknown: list[str], suggestions, removed_by_restrictions: int, explanation: Explanation, trace: list[TraceStep], message: str | None)`
  - `PantryAgent` ("Despensa"), `RestrictionsAgent` ("Restrições"), `RecipeAgent` ("Receitas": recuperação + ranking), `ExplanationAgent` ("Explicação": LLM ou regras) — cada um com `name` e `run`
  - `Orchestrator(recipes: list[Recipe], vector: VectorIndex, llm: LLMClient | None)`: `handle(request) -> Result`; atributos `recipes`, `lexical`, `llm`
  - `load_orchestrator() -> Orchestrator` — lê `config.RECIPES`, abre `config.INDEX_DIR` com `FastEmbedder`, chama `get_llm()`; levanta `agents.DataNotReady(mensagem)` se a base ou o índice não existirem ou se o índice for de outra versão da base

Candidatos = 40 mais próximos na busca semântica ∪ 40 com mais chaves em comum; a similaridade de todos vem de `VectorIndex.similarities`. Consulta semântica: `"{preferência}. Receita com: {nomes dos itens, do mais urgente ao menos}"`. Com preferência preenchida usa `Weights.for_preference()`.

- [ ] **Step 1: Testes que falham**, com 5 receitas, `VectorIndex` real em `tmp_path` com `FakeEmbedder` e LLM falso:
  - fluxo completo: itens {ovo vence hoje, tomate, "leite"}, restrição "amendoim" → sugestões não vazias; nenhuma contém amendoim; `removed_by_restrictions == 1`; rastro com os agentes na ordem `["Despensa", "Restrições", "Receitas", "Explicação"]`; `explanation.source == "llm"`
  - despensa vazia → `suggestions == []`, `message` não vazia, LLM não chamado (Review Focus 1)
  - todos os itens vencidos → `suggestions == []`, `len(expired) == n`, `message` fala em vencidos
  - item "xyzzy" → aparece em `unknown` e o fluxo continua (Review Focus 4)
  - item vencido não aparece em `used` de nenhuma sugestão
  - `assume_staples=False` → nenhuma sugestão tem `staples`
  - `load_orchestrator()` com caminhos de `config` apontados para `tmp_path` vazio → `DataNotReady`
- [ ] **Step 2:** pytest → falha. **Step 3:** implementar. **Step 4:** pytest → passa.

### Task 8: Avaliação

**Files:**
- Create: `src/zerodesperdicio/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `Orchestrator`, `VectorIndex`, `LexicalIndex`, `LLMClient`.
- Produces:
  - `known_item_eval(orchestrator, *, n: int = 300, seed: int = 42, keep: float = 0.7, ks=(1, 5, 10)) -> dict` — para cada receita sorteada (≥ 4 chaves não básicas) monta uma despensa com 70% dos ingredientes dela e mede se a própria receita volta. Modos `semantic`, `lexical`, `hybrid`; por modo: `recall@1`, `recall@5`, `recall@10`, `mrr`. Inclui `n`, `seed`, `corpus_size`.
  - `llm_only_recipes(llm, pantry_names: list[str], n: int = 3) -> list[dict]` — pede ao LLM, sem recuperação, receitas em JSON (`titulo`, `ingredientes`); devolve `[]` se a resposta não for JSON válido
  - `outside_pantry_rate(ingredient_names: list[str], pantry_keys: set[str], staples: frozenset[str]) -> float`
  - CLI: `python -m zerodesperdicio.evaluation retrieval` grava `reports/retrieval_eval.json`

- [ ] **Step 1: Testes que falham**:
  - `known_item_eval` sobre 12 receitas sintéticas bem distintas, com `FakeEmbedder`, `n=8`: resultado tem os três modos e as quatro métricas; `recall@10 == 1.0` no modo `lexical`; todos os valores entre 0 e 1; duas execuções com a mesma semente dão o mesmo dicionário
  - `outside_pantry_rate(["ovos", "tomate", "bacon", "sal"], {"ovo", "tomate"}, STAPLES) == 0.25`; lista vazia → 0.0
  - `llm_only_recipes` com LLM falso devolvendo JSON cercado de texto → lista com os itens; devolvendo `"não sei"` → `[]`
- [ ] **Step 2:** pytest → falha. **Step 3:** implementar. **Step 4:** pytest → passa.
- [ ] **Step 5: Rodar de verdade** — `.venv/bin/python -I -m zerodesperdicio.evaluation retrieval`. Registrar os números no README.

### Task 9: Interface, README e ADRs

**Files:**
- Create: `app/streamlit_app.py`, `README.md`, `docs/adr/0001-recuperacao-hibrida-sem-pytorch.md`, `docs/adr/0002-regras-deterministicas-llm-so-explica.md`, `docs/adr/0003-fonte-de-dados-wikilivros.md`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `load_orchestrator`, `DataNotReady`, `Request`, `PantryStore`, `config`, `evaluation`.

Tela (carregar a skill `dataviz` antes do primeiro gráfico ou cartão de métrica):
- Barra lateral: estado da base, do índice e do LLM; nº de sugestões; "considerar itens básicos"; limites do sistema.
- Aba "Despensa e sugestões": formulário (nome, quantidade, validade), botão "Carregar exemplo", lista por prioridade com status, ações "Usei"/"Descartei", aviso de vencidos; preferência e restrições; botão "Sugerir receitas"; explicação (com selo "LLM" ou "regras"); cartão por receita com cobertura, itens que vencem, faltantes, ingredientes, preparo, link da fonte, data da revisão e licença; rastro dos agentes.
- Aba "LLM × RAG": compara receitas do LLM sem recuperação com as do sistema; sem credencial, explica como ligar.
- Aba "Dados e métricas": funil do pipeline, ingredientes mais frequentes, métricas de recuperação.
- Texto vindo da base passa por `md_escape` antes de ir para `st.markdown`.

- [ ] **Step 1: Testes que falham** em `tests/test_app.py` com `streamlit.testing.v1.AppTest`:
  - caminhos de `config` apontados para `tmp_path` vazio → sem exceção (`not at.exception`) e algum bloco de código na tela contém `zerodesperdicio.pipeline` (Review Focus 2)
  - com base e índice reais (pulado se não existirem): carrega sem exceção; clicar "Carregar exemplo" e depois "Sugerir receitas" → sem exceção e aparece pelo menos um link `pt.wikibooks.org`
  - `md_escape("R$ 5 *a* [x](y) <b>")` não deixa `$`, `*`, `[` nem `<` sem barra invertida
- [ ] **Step 2:** pytest → falha. **Step 3:** implementar o app. **Step 4:** pytest → passa.
- [ ] **Step 5: README** — o que é, como rodar (`make setup data index app`), o que já funciona e o que falta, métricas medidas, fonte e licença dos dados, como ligar o LLM. **ADRs** curtos, status "Proposto".
- [ ] **Step 6: Verificação final** — suíte completa; `streamlit run` em modo headless respondendo 200 em `/_stcore/health`; uma rodada do fluxo de exemplo com a saída conferida a olho.
