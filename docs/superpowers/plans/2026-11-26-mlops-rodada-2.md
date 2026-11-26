# MLOps rodada 2: correções do code review · Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fechar as falhas que o code review da primeira rodada encontrou. Toda configuração que vai para produção passa a ser medida. O portão só compara versões avaliadas nos mesmos dados. A marca `producao` só muda quando tudo deu certo. O app consulta com o modelo de embeddings da configuração.

**Architecture:** Três tarefas independentes, em arquivos disjuntos, feitas em paralelo:
- R1 trata do portão e do registro;
- R2 trata da avaliação, do adaptador `zd-rag-receitas`, do orquestrador e do app;
- R3 trata do CLI, do Makefile e da documentação.

Os contratos entre elas estão nos blocos **Interfaces**. Quem consome uma interface de outra tarefa testa com monkeypatch e não importa código que ainda não existe.

**Tech Stack:** o mesmo da rodada 1 (MLflow 3.18, PyYAML, pytest, Streamlit).

**Spec:** `docs/superpowers/specs/2026-11-19-mlops-design.md`, seção "Portão de promoção", **Regras do portão** 1 a 7 (revisão de 26/11). Este plano implementa essas regras.

## Global Constraints

- Valem todas as restrições da rodada 1: `.superpowers/sdd/2026-11-26-mlops-registro-portao/global-constraints.md`.
- Nenhum comportamento da rodada 1 coberto por teste pode regredir. Um teste só muda quando a regra nova o contradiz, e a mudança vai explicada no relatório.
- Nomes novos são fixos: chave `comparar` no `gates.yaml`; métricas `pref_hybrid_*`; ação de log `promocao_conjunto_novo`; flag `--conjunto-novo` no CLI e `CONJUNTO_NOVO=1` no make; subcomando `uri`.

## Review Focus

1. **Base trocada entre produção e candidata:** o portão recusa a comparação, e só `CONJUNTO_NOVO=1` com motivo promove (R1 `test_gate_different_data_refuses`, R2 `test_eval_set_id_includes_data`).
2. **Pesos de preferência alterados com as outras métricas iguais:** a candidata é recusada se `pref_*` piorar além da tolerância (R1 `test_gate_comparar_tolerance`, R2 `test_pref_pass_uses_preference_weights`).
3. **Falha no meio da promoção** (tag, marca ou log): a produção continua a anterior, sem tags órfãs (R1 `test_promote_tag_failure_keeps_production`).
4. **Índice gerado com outro modelo:** o app trata como dados não prontos, em vez de misturar vetores (R2 `test_load_orchestrator_rejects_index_model_mismatch`).
5. **`make mlflow-ui` com `ZD_HOME` ou `MLFLOW_TRACKING_URI`:** abre o mesmo registro que o CLI usa (R3 `test_mlflow_ui_uses_registry_uri`).

---

### Task R1: Portão e registro

**Files:**
- Modify: `src/zerodesperdicio/mlops/gates.py`
- Modify: `src/zerodesperdicio/mlops/registry.py`
- Modify: `mlops/gates.yaml`
- Test: `tests/test_gates.py`, `tests/test_registry.py` (testes novos; ajuste os testes do QA só quando uma regra nova os contradiz)

**Interfaces:**
- Produces:
  - `GateSpec(minimos, maximos, tolerancia, comparar: tuple[str, ...] = ())`
  - `check_gate(spec, candidate, production, *, candidate_set, production_set, new_set: bool = False) -> GateResult`
  - `registry.promote(model, version, *, by, reason="", gates=None, new_set: bool = False) -> GateResult`
- `gates.yaml`, em `zd-rag-receitas`: `comparar: [pref_hybrid_recall_at_10, pref_hybrid_mrr]`.

Regras (spec, Regras 2, 3, 4, 5 e 6):
- `load_gates`:
  - aceita `comparar`, uma lista de strings não vazias, sem repetição;
  - recusa com `ValueError` todo limite (`minimos`/`maximos`) que não seja finito;
  - recusa `tolerancia` que não seja finita ou que seja < 0;
  - continua exigindo ao menos um item em `minimos` ou `maximos`.
- `check_gate`:
  - **Métricas de `comparar`:** precisam existir na candidata e ser finitas; se faltarem ou não forem finitas, a candidata é recusada. Com produção no mesmo conjunto, recusa se a candidata for menor que `prod − tolerancia` (maior é melhor). Se a produção não tiver a métrica, a comparação é ignorada, como nos mínimos.
  - **Produção sem conjunto** (`production` presente e `production_set is None`): recusa com a razão "a versão em produção não identifica o conjunto de avaliação". Isso corrige o achado sobre a falta da tag.
  - **`new_set=True`:** não checa o conjunto nem compara com a produção; valem só os mínimos e máximos absolutos, e as métricas de `comparar` precisam existir e ser finitas.
- `promote`:
  - com `new_set=True` e `reason.strip()` vazio, lança `RegistryError("Informe o motivo da promoção com conjunto novo")`;
  - quando aprovado com `new_set=True`, a ação no log é `"promocao_conjunto_novo"`.
- `_set_alias`:
  - grava as tags primeiro e move a marca depois;
  - se qualquer passo falhar, remove as tags gravadas (`delete_model_version_tag`, best effort), devolve a marca à versão anterior se já tiver sido movida, e lança `RegistryError`.
- `_log_or_undo`: ao desfazer, remove também as tags que a operação gravou.
- Caminho da recusa: um `OSError` em `_append_log` não sobe. A razão `"não foi possível gravar o log: <erro>"` entra em `razoes`, e o `GateResult` recusado é devolvido.
- `log_candidate`: em qualquer falha depois de `create_run`, chama `client.set_terminated(run_id, "FAILED")` (best effort) antes de lançar `RegistryError`.

- [ ] **Step 1: Escrever os testes que falham**:
  - `test_gates.py`:
    - `test_load_gates_comparar_ok`
    - `test_load_gates_rejects_non_finite_limits` (parametrizado: `.nan`/`.inf` em mínimo e máximo)
    - `test_load_gates_rejects_bad_tolerance` (`-0.1`, `.inf`, `.nan`)
    - `test_load_gates_rejects_bad_comparar` (não lista, string vazia, repetida)
    - `test_gate_comparar_tolerance` (prod 0.50; candidata 0.48 passa e 0.47 recusa, com tolerância 0.02)
    - `test_gate_comparar_missing_or_nan_refuses`
    - `test_gate_production_without_set_refuses`
    - `test_gate_different_data_refuses` (sets `…:dados=aaa` × `…:dados=bbb`)
    - `test_gate_new_set_only_absolute` (pior que a produção, acima dos mínimos e com sets diferentes: aprovado; abaixo do mínimo: recusado)
    - `test_repo_gates_rag_comparar` (o `gates.yaml` real tem os dois `pref_*`)
  - `test_registry.py`:
    - `test_promote_new_set_requires_reason`
    - `test_promote_new_set_logs_action`
    - `test_promote_tag_failure_keeps_production` (monkeypatch `set_model_version_tag` lança; a produção continua a anterior e a versão candidata fica sem tags `promovida_*`)
    - `test_promote_alias_failure_removes_tags`
    - `test_log_failure_undo_removes_tags`
    - `test_refused_log_failure_returns_reasons` (OSError no append do caminho recusado: `aprovado` False e `"não foi possível gravar o log"` em `razoes`)
    - `test_log_candidate_failure_marks_run_failed` (monkeypatch `create_model_version` lança; o run fica `FAILED`)
- [ ] **Step 2: Rodar e ver falhar**: `.venv/bin/python -m pytest tests/test_gates.py tests/test_registry.py -v`
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Rodar a suíte inteira e ver passar**: `.venv/bin/python -m pytest`
- [ ] **Step 5: Commit**: `fix(mlops): portão mede comparar, exige mesmos dados e limites válidos; marca só muda com tudo gravado`

---

### Task R2: Avaliação, `zd-rag-receitas`, orquestrador e app

**Files:**
- Modify: `src/zerodesperdicio/evaluation.py`, `src/zerodesperdicio/mlops/rag.py`, `src/zerodesperdicio/agents.py`, `src/zerodesperdicio/app_support.py`
- Test: `tests/test_evaluation.py`, `tests/test_mlops_rag.py`, `tests/test_data_ready.py`, `tests/test_app.py` (testes novos; ajuste os do QA só se a regra nova os contradiz)

**Interfaces:**
- Consumes: `gates.yaml` com a chave `comparar` (da R1). Nos testes, leia o YAML cru com `yaml.safe_load(config.gates_file().read_text())`, sem `load_gates`, porque a R1 está em paralelo.
- Produces:
  - `known_item_eval(...)`: o resultado ganha a chave `"pref_hybrid"` com as mesmas métricas de `"hybrid"` (`recall@1/5/10`, `mrr`). Na passada com preferência, a mesma receita e os mesmos ingredientes escolhidos são reavaliados com `Request(preference=f"Quero uma receita com {chosen[0]}")`. As chaves `semantic`, `lexical` e `hybrid` precisam continuar com **os mesmos valores de antes**: não consuma o `rng` a mais.
  - `metrics_from_eval(result)`: inclui `pref_hybrid`, totalizando 16 métricas.
  - `eval_set_id(result: dict, recipes_sha256: str) -> str` devolve `known_item:n={n}:seed={seed}:keep={keep}:dados={recipes_sha256[:12]}`. `register_candidate` usa essa função.
  - `load_orchestrator(rag: RagConfig | None = None, *, manifest: dict | None = None) -> Orchestrator`:
    - se `manifest is None`, chama `check_data_ready()`;
    - constrói o embedder com `FastEmbedder(rag.embedding_model)`;
    - se `manifest.get("model")` existe e difere de `rag.embedding_model`, lança `DataNotReady("O índice foi gerado com {m}, mas a configuração usa {e}; gere o índice de novo.")`.
  - `production_config(index_model, *, recipes_sha256: str | None = None, model=MODEL)`: se `recipes_sha256` é dado e difere do parâmetro `dados_sha256` da versão em produção, **mantém** a configuração da produção e devolve `ProductionInfo(n, "A versão vN de zd-rag-receitas foi avaliada com outra base de receitas; rode make eval e promova com CONJUNTO_NOVO=1 para atualizar.")`.
  - `build_orchestrator()`:
    - passa `manifest.get("model")` e `manifest.get("recipes_sha256")` a `production_config`;
    - passa `manifest` a `load_orchestrator`, para não refazer a checagem nem o hash.
  - `production_message(info)`: com versão **e** aviso, devolve `("warning", f"Recuperação: zd-rag-receitas v{N} (marca producao). {aviso}")`.
- `evaluation.main`: se o manifesto existe sem a chave `model`, a mensagem diz "o índice não registra o modelo de embeddings; gere o índice de novo", em vez de "gerado com None".

- [ ] **Step 1: Escrever os testes que falham**:
  - `test_evaluation.py`:
    - `test_known_item_pref_pass` (chave `pref_hybrid` com 4 métricas no intervalo [0, 1])
    - `test_known_item_old_modes_unchanged` (os valores de semantic/lexical/hybrid, com a fixture e a semente, são iguais aos calculados antes da mudança; grave-os no teste)
    - `test_pref_pass_uses_preference_weights` (com `RagConfig(preference_weights=...)` diferente, `pref_hybrid` muda e `hybrid` não)
  - `test_mlops_rag.py`:
    - `test_metrics_include_pref` (16 chaves)
    - `test_eval_set_id_includes_data`
    - `test_metric_names_cover_gate_minimos_and_comparar` (YAML cru)
    - `test_production_config_other_base_warns_keeps_config`
  - `test_data_ready.py`:
    - `test_load_orchestrator_rejects_index_model_mismatch`
    - `test_load_orchestrator_uses_config_embedding_model` (monkeypatch de `FastEmbedder` captura o nome)
    - `test_load_orchestrator_skips_check_with_manifest`
  - `test_app.py`:
    - `test_build_orchestrator_passes_index_model_and_hash` (recupera a asserção perdida na rodada 1)
    - `test_production_message_version_with_warning`
- [ ] **Step 2: Rodar e ver falhar**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Rodar a suíte inteira e ver passar**
- [ ] **Step 5: Commit**: `fix(rag): pesos de preferência medidos; conjunto inclui a base; embedder da configuração`

---

### Task R3: CLI, Makefile e documentação

**Files:**
- Modify: `src/zerodesperdicio/mlops/__main__.py`, `Makefile`, `README.md`, `tests/test_docs.py`
- Test: `tests/test_mlops_cli.py`, `tests/test_docs.py`

**Interfaces:**
- Consumes:
  - `registry.promote(..., new_set: bool = False)`, da R1. Nos testes, faça monkeypatch de `registry.promote` capturando os argumentos, porque a R1 está em paralelo.
  - `registry.tracking_uri()`, que já existe.
- Produces:
  - `promote --conjunto-novo`: repassa `new_set=True`.
  - `uri`: subcomando que imprime `registry.tracking_uri()` e sai com 0.
  - Makefile:
    - **promote**: `CONJUNTO_NOVO` exportado como as outras variáveis; a receita acrescenta `--conjunto-novo` quando `CONJUNTO_NOVO` não é vazio (`$${CONJUNTO_NOVO:+--conjunto-novo}`).
    - **mlflow-ui**: obtém a URI com `$(PY) -I -m zerodesperdicio.mlops uri`. Se a URI começa com `http`, imprime "O registro é um servidor: abra <URI> no navegador." e sai com 0. Senão, roda `.venv/bin/mlflow ui --backend-store-uri "<URI>" --port 5000`.
- README:
  - documentar `CONJUNTO_NOVO=1` (quando usar, motivo obrigatório, ação no log);
  - documentar as métricas `pref_*` e a chave `comparar`;
  - registrar que o `mlflow-ui` segue `ZD_HOME` e `MLFLOW_TRACKING_URI`;
  - registrar que a avaliação usa o embedding da configuração.
- `tests/test_docs.py`, pontos menores adiados da rodada 1:
  - o extrator de `make <alvo>` só aceita `make` no início de um comando, seja numa linha de bloco de código, seja logo depois de uma crase de abertura; ignora `# make …`;
  - `_log_keys` confere que **todos** os dicionários com `quando` em `registry.py` têm o mesmo conjunto de chaves;
  - `git check-ignore` com código 128 (fora de um repositório) leva a `pytest.skip`;
  - as métricas do `gates.yaml` (`minimos`, `maximos` e `comparar`) precisam aparecer também no spec.

- [ ] **Step 1: Escrever os testes que falham**:
  - `test_mlops_cli.py`:
    - `test_cli_promote_conjunto_novo_passes_new_set`
    - `test_cli_promote_default_new_set_false`
    - `test_cli_uri_prints_tracking_uri`
    - `test_makefile_promote_conjunto_novo` (`make -n` com e sem a variável)
    - `test_mlflow_ui_uses_registry_uri` (make real com `PY` trocado por um script que imprime uma URI http; a saída contém "abra")
  - `test_docs.py`: os quatro itens acima, mais `CONJUNTO_NOVO` e `comparar` presentes no README.
- [ ] **Step 2: Rodar e ver falhar**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Rodar a suíte inteira e ver passar**. A seção do spec sobre comparar já foi escrita pelo controlador; não a altere.
- [ ] **Step 5: Commit**: `fix(cli): promoção com conjunto novo, mlflow-ui segue o registro; docs atualizadas`
