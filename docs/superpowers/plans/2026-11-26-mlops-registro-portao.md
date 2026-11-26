# MLOps — registro de versões, portão de promoção e volta atrás · Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Toda avaliação do RAG de receitas vira uma versão candidata no registro do MLflow; `make promote` só move a marca `producao` se o portão aprovar, `make rollback` devolve a marca com motivo, e o app carrega a configuração marcada sem nunca parar por causa do MLflow.

**Architecture:** Um pacote `zerodesperdicio.mlops` com três camadas: `gates.py` (portão puro, sem MLflow), `registry.py` (único lugar que fala com o MLflow: candidatas, marca, log de promoções) e `rag.py` (adaptador do `zd-rag-receitas`). A configuração da recuperação deixa de ser constante e vira um `RagConfig` injetado no orquestrador. O `zd-frescor` (Vitor) e o `zd-explicacao` (LLMOps, próximo plano) usam a mesma interface do registro; aqui só ganham limites no `gates.yaml`.

**Tech Stack:** MLflow 3.18 (registro com aliases, backend SQLite local), PyYAML, pytest, Streamlit.

**Spec:** `docs/superpowers/specs/2026-11-19-mlops-design.md` (ADR 0007). Este plano cobre as seções "Modelos registrados" e "Portão de promoção". DVC, CI, LLMOps/monitoramento e Docker ficam para os próximos planos.

## Global Constraints

- Python ≥ 3.10, só CPU; dependências novas: `mlflow>=3.18,<4` e `pyyaml>=6` (dependências principais, porque o app lê o registro).
- Testes sem rede e sem chave de API (anti-objetivo do spec). Testes do MLflow usam SQLite em pasta temporária.
- O app nunca para por causa do MLflow: registro vazio, fora do ar ou incompatível → configuração padrão do código + aviso na tela.
- Limites do portão só em `mlops/gates.yaml` (versionado). Valores do spec: `zd-frescor` F1 macro ≥ 0,85 e recall de "estragado" ≥ 0,85, tolerância 0,01; `zd-rag-receitas` recall@10 ≥ 0,80 e MRR ≥ 0,45, tolerância 0,02; `zd-explicacao` citações válidas ≥ 0,95, afirmações de "seguro" = 0, tolerância 0,02.
- Marca do registro: `producao` (sem acento). Toda promoção, recusa e volta atrás vai para as tags da versão e para `reports/promocoes.jsonl`.
- Nada de `mlruns/`, banco do MLflow, dados ou modelos no git.
- Padrão do código: identificadores em inglês, mensagens, docstrings e chaves de arquivo de configuração em português, `from __future__ import annotations`, caminhos via `config.*()` (nunca relativos à pasta atual).
- Nomes de métrica do MLflow não aceitam `@`: `recall@10` vira `recall_at_10`.

## Review Focus

1. **MLflow fora do ar ou lento na abertura do app**: o app abre em segundos com a configuração padrão e mostra um aviso, sem travar nos retries do cliente HTTP. Teste: `test_production_config_unreachable_falls_back` (Task 4).
2. **Candidata sem uma métrica exigida pelo portão** (ex.: o treino do Vitor registra `f1_estragado`, mas o portão pede `recall_estragado`): é recusada com o nome da métrica, sem `KeyError`. Teste: `test_gate_missing_metric_refuses` (Task 1).
3. **Versão, modelo ou motivo inválido em `promote`/`rollback`**: aparece um erro claro em português e a marca não muda. Testes: `test_rollback_requires_reason`, `test_promote_unknown_version` (Task 3) e `test_cli_error_exit_code` (Task 5).
4. **Comando rodado de outra pasta** (ex.: `cd app && python -m ...`): os artefatos vão para `<raiz>/mlruns`, nunca para `./mlruns` da pasta atual (o padrão do MLflow). Teste: `test_artifacts_not_in_cwd` (Task 3).
5. **Conjunto de avaliação diferente do da produção** (outra semente, outro n, outro conjunto de teste de fotos): o portão recusa em vez de comparar números incomparáveis. Teste: `test_gate_different_eval_set_refuses` (Task 1).

---

### Task 1: Portão de promoção (puro, sem MLflow)

**Files:**
- Create: `mlops/gates.yaml`
- Create: `src/zerodesperdicio/mlops/__init__.py` (vazio, com docstring do pacote)
- Create: `src/zerodesperdicio/mlops/gates.py`
- Modify: `src/zerodesperdicio/config.py` (adicionar `gates_file()`)
- Modify: `pyproject.toml`, `uv.lock` (via `uv add "mlflow>=3.18,<4" "pyyaml>=6"`)
- Test: `tests/test_gates.py`

**Interfaces:**
- Produces:
  - `config.gates_file() -> Path`: `_REPO_ROOT / "mlops" / "gates.yaml"`. O portão é código versionado, então **não** segue `ZD_HOME`.
  - `@dataclass(frozen=True) GateSpec(minimos: dict[str, float], maximos: dict[str, float], tolerancia: float)`
  - `@dataclass(frozen=True) GateResult(aprovado: bool, razoes: list[str])`
  - `load_gates(path: Path | None = None) -> dict[str, GateSpec]`: chaves `minimos`/`maximos` opcionais (padrão `{}`); `ValueError` se faltar `tolerancia` ou algum valor não for número.
  - `check_gate(spec: GateSpec, candidate: dict[str, float], production: dict[str, float] | None, *, candidate_set: str, production_set: str | None) -> GateResult`

`mlops/gates.yaml`:

```yaml
# Portão de promoção (ADR 0007). Mudar um limite exige ADR.
# minimos: maior é melhor; maximos: menor é melhor; tolerancia: quanto pode piorar contra a producao.
zd-rag-receitas:
  minimos: {hybrid_recall_at_10: 0.80, hybrid_mrr: 0.45}
  tolerancia: 0.02
zd-frescor:
  minimos: {f1_macro: 0.85, recall_estragado: 0.85}
  tolerancia: 0.01
zd-explicacao:
  minimos: {citacoes_validas: 0.95}
  maximos: {afirmacoes_seguro: 0}
  tolerancia: 0.02
```

Regras do `check_gate`, avaliadas todas para que `razoes` liste cada falha:
- métrica de `minimos`/`maximos` ausente na candidata → recusa (`"falta a métrica X"`);
- abaixo do mínimo ou acima do máximo → recusa;
- havendo produção: se `production_set` existe e é diferente de `candidate_set` → recusa (`"conjunto de avaliação diferente do da produção"`), sem comparar números; senão, mínimo pior que `prod − tolerancia` ou máximo pior que `prod + tolerancia` → recusa. Métrica ausente na produção → não compara essa métrica;
- `production is None` (primeira promoção) → só os limites absolutos.

- [ ] **Step 1: Instalar dependências e criar `mlops/gates.yaml`**

Run: `uv add "mlflow>=3.18,<4" "pyyaml>=6" && .venv/bin/python -c "import mlflow, yaml; print(mlflow.__version__)"`
Expected: `3.18.x`

- [ ] **Step 2: Escrever os testes que falham** em `tests/test_gates.py`

```python
SPEC = GateSpec(minimos={"recall": 0.80, "mrr": 0.45}, maximos={"seguro": 0}, tolerancia=0.02)
OK = {"recall": 0.81, "mrr": 0.49, "seguro": 0}

def test_gate_first_promotion_only_minimums():
    assert check_gate(SPEC, OK, None, candidate_set="a", production_set=None).aprovado

def test_gate_below_minimum_refuses():
    r = check_gate(SPEC, {**OK, "recall": 0.79}, None, candidate_set="a", production_set=None)
    assert not r.aprovado and any("recall" in x for x in r.razoes)

def test_gate_above_maximum_refuses():
    assert not check_gate(SPEC, {**OK, "seguro": 1}, None, candidate_set="a", production_set=None).aprovado

def test_gate_tolerance_against_production():
    prod = {"recall": 0.85, "mrr": 0.49, "seguro": 0}
    assert check_gate(SPEC, {**OK, "recall": 0.83}, prod, candidate_set="a", production_set="a").aprovado
    r = check_gate(SPEC, {**OK, "recall": 0.82}, prod, candidate_set="a", production_set="a")
    assert not r.aprovado and any("produção" in x for x in r.razoes)

def test_gate_missing_metric_refuses():
    r = check_gate(SPEC, {"recall": 0.9, "seguro": 0}, None, candidate_set="a", production_set=None)
    assert not r.aprovado and any("mrr" in x for x in r.razoes)

def test_gate_different_eval_set_refuses():
    r = check_gate(SPEC, OK, OK, candidate_set="a", production_set="b")
    assert not r.aprovado and any("conjunto de avaliação" in x for x in r.razoes)

def test_load_gates_repo_file():
    g = load_gates()
    assert g["zd-rag-receitas"].minimos == {"hybrid_recall_at_10": 0.80, "hybrid_mrr": 0.45}
    assert g["zd-frescor"].tolerancia == 0.01 and g["zd-explicacao"].maximos == {"afirmacoes_seguro": 0}

def test_load_gates_rejects_missing_tolerance(tmp_path):
    p = tmp_path / "g.yaml"; p.write_text("m: {minimos: {a: 1}}")
    with pytest.raises(ValueError):
        load_gates(p)
```

Para `0.85 − 0.02` não falhar por arredondamento de ponto flutuante (0.83 tem que passar), compare com `round(..., 6)`.

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_gates.py -v`
Expected: FAIL (`ModuleNotFoundError: zerodesperdicio.mlops`)

- [ ] **Step 4: Implementar `config.gates_file`, `load_gates` e `check_gate`**

`load_gates` usa `yaml.safe_load`.

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/bin/python -m pytest tests/test_gates.py -v && .venv/bin/python -m pytest`
Expected: todos passam (117 + 8)

- [ ] **Step 6: Commit**

```bash
git add mlops/gates.yaml src/zerodesperdicio/mlops src/zerodesperdicio/config.py tests/test_gates.py pyproject.toml uv.lock
git commit -m "feat(mlops): portão de promoção com limites em mlops/gates.yaml"
```

---

### Task 2: Configuração da recuperação injetável (`RagConfig`)

Hoje os pesos e o número de candidatas são constantes em `agents.py`/`ranking.py`. Para o app usar a versão marcada, a configuração precisa entrar pelo construtor. O comportamento não muda.

**Files:**
- Create: `src/zerodesperdicio/rag_config.py`
- Modify: `src/zerodesperdicio/agents.py` (remover `CANDIDATES_PER_SOURCE`; `RecipeAgent`, `Orchestrator` e `load_orchestrator` recebem `rag`)
- Test: `tests/test_rag_config.py`

**Interfaces:**
- Consumes: `ranking.Weights`, `config.EMBEDDING_MODEL`
- Produces:
  - `@dataclass(frozen=True) RagConfig(embedding_model: str = config.EMBEDDING_MODEL, weights: Weights = Weights(), preference_weights: Weights = Weights.for_preference(), candidates_per_source: int = 40)`
  - `RagConfig.to_dict() -> dict` / `RagConfig.from_dict(d: dict) -> RagConfig` (formato do artefato `rag_config.json`: `{"embedding_model", "weights": {"coverage","urgency","semantic"}, "preference_weights": {...}, "candidates_per_source"}`); `from_dict` lança `ValueError` com chave desconhecida ou faltando, e se `candidates_per_source < 1`
  - `RagConfig.to_params() -> dict[str, str]`: parâmetros planos para o MLflow (`embedding_model`, `w_cobertura`, `w_urgencia`, `w_semantica`, `wp_cobertura`, `wp_urgencia`, `wp_semantica`, `candidatas_por_fonte`)
  - `RecipeAgent(recipes, vector, lexical, rag: RagConfig = RagConfig())`: usa `rag.candidates_per_source`, `rag.weights` e `rag.preference_weights` (no lugar de `Weights.for_preference()`)
  - `Orchestrator(recipes, vector, llm, rag: RagConfig = RagConfig())`, com o atributo `self.rag`
  - `load_orchestrator(rag: RagConfig | None = None) -> Orchestrator`

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_rag_config.py`

```python
def test_roundtrip():
    c = RagConfig(weights=Weights(0.6, 0.3, 0.1), candidates_per_source=25)
    assert RagConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c

def test_from_dict_rejects_bad_input():
    d = RagConfig().to_dict()
    for bad in ({**d, "extra": 1}, {k: v for k, v in d.items() if k != "weights"}, {**d, "candidates_per_source": 0}):
        with pytest.raises(ValueError):
            RagConfig.from_dict(bad)

def test_params_are_flat_strings():
    p = RagConfig().to_params()
    assert p["w_cobertura"] == "0.5" and p["wp_semantica"] == "0.3" and p["candidatas_por_fonte"] == "40"

def test_orchestrator_uses_injected_config(tmp_path):
    # mesmo padrão de fixture de tests/test_evaluation.py (FakeEmbedder + make_recipe)
    ...
    orch = Orchestrator(recipes, vec, None, RagConfig(candidates_per_source=1))
    res = orch.handle(Request(items=[PantryItem(k) for k in recipes[0].ingredient_keys], today=date(2026, 1, 1)))
    step = next(s for s in res.trace if s.action == "recuperação híbrida")
    assert step.detail.startswith("1 pela busca semântica")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_rag_config.py -v`
Expected: FAIL (`ModuleNotFoundError: zerodesperdicio.rag_config`)

- [ ] **Step 3: Implementar `rag_config.py` e ligar em `agents.py`**

- [ ] **Step 4: Rodar a suíte inteira** (o comportamento padrão não pode mudar)

Run: `.venv/bin/python -m pytest`
Expected: todos passam

- [ ] **Step 5: Commit**

```bash
git add src/zerodesperdicio/rag_config.py src/zerodesperdicio/agents.py tests/test_rag_config.py
git commit -m "refactor: configuração da recuperação injetável (RagConfig)"
```

---

### Task 3: Registro de versões, promoção e volta atrás

**Files:**
- Create: `src/zerodesperdicio/mlops/registry.py`
- Modify: `tests/conftest.py` (fixture do MLflow)
- Modify: `.gitignore` (`mlruns/`)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `gates.load_gates`, `gates.check_gate`, `GateResult`, `config.root()`, `config.reports_dir()`
- Produces:
  - `ALIAS = "producao"`
  - `class RegistryError(Exception)`: mensagem em português, pronta para a tela/terminal
  - `@dataclass(frozen=True) Version(model: str, version: int, run_id: str, metrics: dict[str, float], params: dict[str, str], tags: dict[str, str])`; `tags` são as do run (`conjunto_avaliacao`, `git_commit`, …)
  - `tracking_uri() -> str`: `MLFLOW_TRACKING_URI` se definida; senão `sqlite:///{config.root()}/mlruns/mlflow.db` (cria a pasta)
  - `log_candidate(model: str, *, params: dict[str, str], metrics: dict[str, float], artifacts: dict[str, dict], eval_set: str, tags: dict[str, str] | None = None) -> Version`
  - `get_version(model: str, version: int) -> Version` (`RegistryError` se o modelo ou a versão não existe)
  - `production(model: str) -> Version | None`
  - `load_artifact(v: Version, name: str) -> dict`
  - `promote(model: str, version: int, *, by: str, reason: str = "", gates: dict[str, GateSpec] | None = None) -> GateResult`
  - `rollback(model: str, version: int, *, by: str, reason: str) -> None`

Decisões fixadas (validadas num protótipo com MLflow 3.18):
- **Cliente:** antes de criar o `MlflowClient(tracking_uri(), tracking_uri())`, aplicar `os.environ.setdefault` para `MLFLOW_HTTP_REQUEST_MAX_RETRIES="0"`, `MLFLOW_HTTP_REQUEST_TIMEOUT="5"` e `MLFLOW_DISABLE_AGENT_HINT="1"`. O `import mlflow` fica **dentro** das funções (leva ~6 s; os demais testes e o app sem registro não pagam esse custo).
- **Experimento = nome do modelo.** Se não existir, criar com `artifact_location`: para URI `sqlite:///<arq>`, usar `(Path(<arq>).parent / "artifacts" / model).as_uri()`; para servidor remoto, `None` (o servidor decide). Sem isso o MLflow grava em `./mlruns` da pasta atual.
- **Candidata:** um run com `log_params`, `log_metrics`, `log_dict(artifact, f"config/{name}")` para cada artefato, mais as tags `conjunto_avaliacao=eval_set` e `git_commit` (`git rev-parse HEAD` na raiz do repo, ou `"desconhecido"` se falhar), além das `tags` extras. Depois `create_registered_model` (se não existir) e `create_model_version(model, source=f"{run.info.artifact_uri}/config", run_id=...)`. Não usar `mlflow.register_model`: a configuração não é um modelo MLflow.
- **Marca:** `set_registered_model_alias`/`get_model_version_by_alias`. Alias ausente e modelo ausente lançam `MlflowException`, que `production` transforma em `None`. Outros erros (rede) sobem como `RegistryError`.
- **promote:** modelo fora do `gates.yaml` → `RegistryError`. Versão já em `producao` → `GateResult(True, ["a versão N já está em produção"])`, sem registrar nada. Caso contrário, roda o portão contra `production(model)`; se aprovar, move a marca e grava as tags da versão `promovida_em`/`promovida_por`/`motivo`. Aprovando ou recusando, acrescenta uma linha ao log.
- **rollback:** `reason.strip()` vazio → `RegistryError`. Versão inexistente → `RegistryError`. Move a marca sem portão e grava as tags `retorno_em`/`retorno_por`/`motivo`, mais uma linha no log.
- **Log** `config.reports_dir() / "promocoes.jsonl"`, uma linha JSON por evento: `{"quando": ISO-8601 UTC, "modelo", "acao": "promocao" | "promocao_recusada" | "retorno", "de": int | null, "para": int, "por", "motivo", "razoes": [...], "metricas": {...}}`.

Fixture em `tests/conftest.py`: um banco SQLite por **sessão** (criar o banco leva ~4 s), e cada teste usa nomes de modelo únicos.

```python
@pytest.fixture(scope="session")
def _mlflow_db(tmp_path_factory):
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"

@pytest.fixture
def registry_env(_mlflow_db, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _mlflow_db)
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    return f"zd-teste-{uuid.uuid4().hex[:8]}"   # nome de modelo exclusivo do teste
```

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_registry.py`

```python
GATES = lambda m: {m: GateSpec(minimos={"recall": 0.80}, maximos={}, tolerancia=0.02)}

def cand(m, recall, eval_set="s1"):
    return registry.log_candidate(m, params={"p": "1"}, metrics={"recall": recall},
                                  artifacts={"cfg.json": {"a": recall}}, eval_set=eval_set)

def test_log_candidate_versions_and_artifact(registry_env):
    m = registry_env
    v1, v2 = cand(m, 0.81), cand(m, 0.82)
    assert (v1.version, v2.version) == (1, 2)
    assert registry.get_version(m, 2).metrics == {"recall": 0.82}
    assert registry.load_artifact(v2, "cfg.json") == {"a": 0.82}
    assert registry.production(m) is None

def test_promote_and_refuse(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85); cand(m, 0.82); cand(m, 0.79)
    assert registry.promote(m, 1, by="pedro", gates=GATES(m)).aprovado
    assert not registry.promote(m, 2, by="pedro", gates=GATES(m)).aprovado  # piora 0,03 > 0,02
    assert not registry.promote(m, 3, by="pedro", gates=GATES(m)).aprovado  # abaixo do mínimo
    assert registry.production(m).version == 1
    log = [json.loads(l) for l in (tmp_path / "reports/promocoes.jsonl").read_text().splitlines()]
    assert [e["acao"] for e in log] == ["promocao", "promocao_recusada", "promocao_recusada"]
    assert log[0]["de"] is None and log[0]["para"] == 1 and log[0]["por"] == "pedro"

def test_promote_same_version_is_noop(registry_env, tmp_path):
    ...  # promove a 1 duas vezes; a segunda devolve aprovado com "já está em produção" e o log tem 1 linha

def test_rollback(registry_env, tmp_path):
    ...  # promove 1, promove 2 (melhor), rollback para 1 com motivo → production == 1; log termina em "retorno" com de=2, para=1

def test_rollback_requires_reason(registry_env):
    ...  # reason="  " → RegistryError; production continua a mesma

def test_promote_unknown_version(registry_env):
    ...  # promote(m, 99) e rollback(m, 99, reason="x") → RegistryError; modelo sem portão → RegistryError

def test_artifacts_not_in_cwd(registry_env, tmp_path, monkeypatch):
    work = tmp_path / "outra_pasta"; work.mkdir(); monkeypatch.chdir(work)
    cand(registry_env, 0.9)
    assert not (work / "mlruns").exists()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v`
Expected: FAIL (`ImportError: cannot import name 'registry'`)

- [ ] **Step 3: Implementar `registry.py` e adicionar `mlruns/` ao `.gitignore`**

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v && .venv/bin/python -m pytest`
Expected: todos passam; `test_registry.py` leva menos de ~30 s

- [ ] **Step 5: Commit**

```bash
git add src/zerodesperdicio/mlops/registry.py tests/conftest.py tests/test_registry.py .gitignore
git commit -m "feat(mlops): registro de versões com marca producao, promoção com portão e volta atrás"
```

---

### Task 4: `zd-rag-receitas` no registro (avaliação → candidata; produção → app)

**Files:**
- Create: `src/zerodesperdicio/mlops/rag.py`
- Modify: `src/zerodesperdicio/evaluation.py` (`main`: opções `--registrar` e `--config`; o orquestrador é carregado com o `RagConfig` escolhido)
- Test: `tests/test_mlops_rag.py`

**Interfaces:**
- Consumes: `registry.log_candidate`, `registry.production`, `registry.load_artifact`, `RegistryError`, `RagConfig`, `known_item_eval` (formato do resultado em `reports/retrieval_eval.json`)
- Produces:
  - `MODEL = "zd-rag-receitas"`
  - `metrics_from_eval(result: dict) -> dict[str, float]`: `{f"{modo}_{k.replace('@', '_at_')}": v}` para `semantic`/`lexical`/`hybrid` (ex.: `hybrid_recall_at_10`, `hybrid_mrr`)
  - `eval_set_id(result: dict) -> str`: `known_item:n={n}:seed={seed}:keep={keep}`. A base fica de fora de propósito: ela muda a cada `make build`, e o hash dela vai no parâmetro `dados_sha256`
  - `register_candidate(result: dict, rag: RagConfig, recipes_sha256: str, *, model: str = MODEL) -> registry.Version`: params = `rag.to_params() | {"dados_sha256": ...}`; artefatos `rag_config.json` e `retrieval_eval.json`
  - `@dataclass(frozen=True) ProductionInfo(version: int | None, warning: str | None)`
  - `production_config(index_model: str | None, *, model: str = MODEL) -> tuple[RagConfig, ProductionInfo]`. Nunca lança exceção. Casos: (a) sem versão `producao` → `RagConfig()` com o aviso "Nenhuma versão de zd-rag-receitas marcada como producao; usando a configuração padrão do código."; (b) `RegistryError` ou qualquer exceção → `RagConfig()` com o aviso "Registro do MLflow indisponível (…); usando a configuração padrão do código."; (c) `embedding_model` da produção diferente de `index_model` → `RagConfig()` com o aviso de que o índice foi gerado com outro modelo; (d) caso normal → configuração da produção, `ProductionInfo(n, None)`
- CLI: `python -m zerodesperdicio.evaluation retrieval [--n N] [--config ARQ.json] [--registrar]`. `--config` lê um `RagConfig.from_dict` para experimentar pesos sem editar código; `--registrar` chama `register_candidate` e imprime `zd-rag-receitas: candidata vN registrada (make promote MODEL=zd-rag-receitas VERSION=N)`.

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_mlops_rag.py`

```python
RESULT = {"semantic": {"recall@1": .06, "recall@5": .13, "recall@10": .19, "mrr": .09},
          "lexical": {"recall@1": .71, "recall@5": .85, "recall@10": .88, "mrr": .77}, "hybrid": {"recall@1": .33, "recall@5": .72, "recall@10": .81, "mrr": .489},
          "n": 300, "seed": 42, "keep": 0.7, "corpus_size": 2088}

def test_metrics_and_eval_set():
    m = rag.metrics_from_eval(RESULT)
    assert m["hybrid_recall_at_10"] == .81 and m["hybrid_mrr"] == .489 and len(m) == 12
    assert rag.eval_set_id(RESULT) == "known_item:n=300:seed=42:keep=0.7"

def test_production_config_roundtrip(registry_env):
    cfg = RagConfig(weights=Weights(0.6, 0.3, 0.1))
    v = rag.register_candidate(RESULT, cfg, "abc", model=registry_env)
    assert v.params["dados_sha256"] == "abc"
    registry.promote(registry_env, v.version, by="t", gates={registry_env: load_gates()["zd-rag-receitas"]})
    got, info = rag.production_config(cfg.embedding_model, model=registry_env)
    assert got == cfg and info == rag.ProductionInfo(v.version, None)

def test_production_config_empty_registry(registry_env):
    got, info = rag.production_config(None, model=registry_env)
    assert got == RagConfig() and info.version is None and "producao" in info.warning

def test_production_config_embedding_mismatch(registry_env):
    ...  # promove uma versão e chama production_config("outro-modelo") → RagConfig() com aviso que menciona o índice

def test_production_config_unreachable_falls_back(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    t = time.monotonic()
    got, info = rag.production_config(None)
    assert got == RagConfig() and "indisponível" in info.warning and time.monotonic() - t < 15
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_mlops_rag.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implementar `mlops/rag.py` e as opções novas em `evaluation.main`**

O `dados_sha256` vem de `hashlib.sha256(config.recipes_path().read_bytes())`, igual ao `load_orchestrator`.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest`
Expected: todos passam

- [ ] **Step 5: Commit**

```bash
git add src/zerodesperdicio/mlops/rag.py src/zerodesperdicio/evaluation.py tests/test_mlops_rag.py
git commit -m "feat(mlops): avaliação do RAG registra candidata zd-rag-receitas; configuração de produção com fallback"
```

---

### Task 5: Comandos `make promote`, `make rollback`, `make status`

**Files:**
- Create: `src/zerodesperdicio/mlops/__main__.py`
- Modify: `Makefile`
- Test: `tests/test_mlops_cli.py`

**Interfaces:**
- Consumes: `registry.promote`, `registry.rollback`, `registry.production`, `RegistryError`, `load_gates`
- Produces: `main(argv: list[str] | None = None) -> int` (código de saída), com os subcomandos:
  - `promote --model M --version N [--motivo TXT] [--por NOME]`: imprime `APROVADO` ou `RECUSADO` com as razões; sai com 0 se aprovado, 1 se recusado
  - `rollback --model M --version N --motivo TXT [--por NOME]`
  - `status`: para cada modelo do `gates.yaml`, imprime `<modelo>: producao = vN (métricas do portão)` ou `<modelo>: nenhuma versão em produção`
  - `--por` por padrão vem de `git config user.name`, e se falhar de `getpass.getuser()`
  - `RegistryError` → mensagem em `stderr`, saída 2

Makefile (as variáveis vazias fazem o argparse reclamar, o que é o comportamento desejado):

```make
eval:
	$(PY) -I -m zerodesperdicio.evaluation retrieval --registrar

promote:
	$(PY) -I -m zerodesperdicio.mlops promote --model "$(MODEL)" --version "$(VERSION)" --motivo "$(MOTIVO)"

rollback:
	$(PY) -I -m zerodesperdicio.mlops rollback --model "$(MODEL)" --version "$(VERSION)" --motivo "$(MOTIVO)"

status:
	$(PY) -I -m zerodesperdicio.mlops status

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Acrescente os alvos novos ao `.PHONY`.

Fixture `gates_for` (no próprio `tests/test_mlops_cli.py`): devolve uma função `gates_for(m: str) -> None` que grava `{m: {minimos: {recall: 0.8}, tolerancia: 0.02}}` em `tmp_path / "gates.yaml"` e faz `monkeypatch.setattr(config, "gates_file", lambda: caminho)`. Por isso `load_gates()` e o CLI precisam chamar `config.gates_file()` na hora de usar, não guardá-lo numa constante na importação.

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_mlops_cli.py` (usam `registry_env` e um `gates.yaml` temporário via `monkeypatch.setattr(config, "gates_file", ...)`)

```python
def test_cli_promote_and_status(registry_env, capsys, gates_for):
    gates_for(registry_env)
    registry.log_candidate(registry_env, params={}, metrics={"recall": .9}, artifacts={}, eval_set="s")
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    assert main(["status"]) == 0 and f"{registry_env}: producao = v1" in capsys.readouterr().out

def test_cli_refused_exit_code(...):        # candidata abaixo do mínimo → 1 e "RECUSADO" na saída
def test_cli_error_exit_code(registry_env, capsys, gates_for):
    assert main(["rollback", "--model", registry_env, "--version", "7", "--motivo", "x"]) == 2
    assert capsys.readouterr().err.strip()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_mlops_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `__main__.py` e o Makefile**

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest && make status`
Expected: os testes passam; `make status` imprime as três linhas "nenhuma versão em produção" (o registro local ainda está vazio)

- [ ] **Step 5: Commit**

```bash
git add src/zerodesperdicio/mlops/__main__.py Makefile tests/test_mlops_cli.py
git commit -m "feat(mlops): comandos make promote/rollback/status e mlflow-ui"
```

---

### Task 6: App carrega a versão `producao`

**Files:**
- Modify: `app/streamlit_app.py` (`get_orchestrator` e a aba "Dados e métricas")
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `rag.production_config`, `ProductionInfo`, `load_orchestrator(rag)`, `index.read_manifest(config.index_dir())["model"]`
- Produces: `get_orchestrator(root: str) -> tuple[Orchestrator, ProductionInfo]` (continua em `st.cache_resource`, então o registro é lido uma vez por processo)

Comportamento: a aba "Dados e métricas" ganha o subtítulo "Versão em produção", logo antes de "Qualidade da recuperação". Com versão: `st.success("Recuperação: zd-rag-receitas v{N} (marca producao).")`. Sem versão: `st.warning(info.warning)`. O rastro do agente Receitas já mostra os pesos usados e fica como está.

- [ ] **Step 1: Escrever o teste que falha** em `tests/test_app.py`: com `ZD_HOME` vazio e `MLFLOW_TRACKING_URI=http://127.0.0.1:9`, o app continua sem exceção e mostra as instruções. Ele protege a ordem: os dados são conferidos antes de qualquer acesso ao registro, para que um registro fora do ar não esconda a mensagem de "gere os dados".

- [ ] **Step 2: Rodar, implementar e rodar de novo**

Run: `.venv/bin/python -m pytest tests/test_app.py -v && .venv/bin/python -m pytest`
Expected: todos passam

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py tests/test_app.py
git commit -m "feat(app): recuperação usa a versão producao do registro, com aviso quando cai no padrão"
```

---

### Task 7: Ponta a ponta com dados reais e documentação

Precisa de rede: `make fetch` leva ~5 min e o índice baixa ~220 MB na primeira vez.

**Files:**
- Modify: `docs/superpowers/specs/2026-11-19-mlops-design.md` (contratos adaptados, conforme a decisão de 26/11)
- Modify: `docs/adr/0007-mlops-registro-portao-e-retorno.md` (consequência: registro local em `mlruns/`, armazenamento compartilhado ainda em aberto)
- Modify: `README.md` (seção "MLOps": `make eval`, `make status`, `make promote`, `make rollback`, `make mlflow-ui`)
- Modify: `docs/roadmap.md` (marcar o item da semana 4 como parcial: registro, portão e volta atrás feitos; DVC no pipeline pendente)

Mudanças no spec, seção "Dados e pipeline":
- **Contrato com o Eduardo:** `datasets/freshness/organized/{produto_fresco,produto_estragado}/<produto>/` + `manifest.csv`; entram só as linhas com `status == accepted`.
- **Contrato com o Vitor:** o treino continua em `zerodesperdicio.vision.train_classifier`. O registro passa a ser `registry.log_candidate("zd-frescor", ..., eval_set=<sha256 do manifesto do conjunto de teste>)` com as métricas `f1_macro` e `recall_estragado`, no lugar de `vision/mlflow_tracking.py` (o portão próprio de 0,80 sai). Imagens e `.joblib` saem do git e vão para o DVC.

- [ ] **Step 1: Rodar o fluxo real**

Run: `make fetch build index && make eval && make status`
Expected: `zd-rag-receitas: candidata v1 registrada`, com métricas próximas de `hybrid_recall_at_10 ≈ 0.81` e `hybrid_mrr ≈ 0.489`

- [ ] **Step 2: Promover, provar o portão e a volta atrás**

Run: `make promote MODEL=zd-rag-receitas VERSION=1 MOTIVO="linha de base v0.1"`, depois uma candidata pior via `--config` com `{"weights": {"coverage": 0.1, "urgency": 0.1, "semantic": 0.8}, ...}` e `make promote` dela
Expected: v1 `APROVADO`; a pior `RECUSADO`, com as razões; `make status` mostra v1; `reports/promocoes.jsonl` tem as duas linhas

- [ ] **Step 3: Conferir o app**

Run: `make app` → a aba "Dados e métricas" mostra "zd-rag-receitas v1"; com `MLFLOW_TRACKING_URI=http://127.0.0.1:9 make app`, mostra o aviso e as sugestões continuam funcionando.

- [ ] **Step 4: Atualizar a documentação e conferir que nada pesado entrou no git**

Run: `git status --short && git check-ignore -v mlruns/mlflow.db`
Expected: só arquivos de documentação modificados; `mlruns/` ignorado

- [ ] **Step 5: Commit**

```bash
git add docs README.md reports/retrieval_eval.json
git commit -m "docs(mlops): registro e portão no README, contratos adaptados no spec e ADR 0007"
```
