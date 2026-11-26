# ZeroDesperdício

Assistente de IA que ajuda a decidir **o que fazer com os alimentos de casa antes que estraguem**: prioriza o que vence primeiro e sugere receitas reais, com justificativa e fonte.

Challenge 2 · Big Idea "Qualidade de Vida". Visão geral do projeto: [ZeroDesperdicio_visao_geral.pdf](ZeroDesperdicio_visao_geral.pdf).

> **Estado: MVP de demonstração (v0.1, 11/11/2026).** Cobre o RAG e o sistema multiagentes (semanas 2 e 3 do roadmap). O registro de versões do RAG no MLflow, com portão de promoção e volta atrás, também está pronto (seção MLOps). Visão computacional, DVC no pipeline, CI e retreino ficam para as próximas semanas.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Python 3.10 ou mais novo.

```bash
make setup    # cria .venv e instala as dependências
make fetch    # baixa o Livro de receitas do Wikilivros (~5 min; respeita o limite da API e é retomável)
make build    # gera data/processed/recipes.jsonl + manifest.json
make index    # gera o índice vetorial (baixa o modelo de embeddings, ~220 MB, na 1ª vez)
make eval     # mede a recuperação -> reports/retrieval_eval.json e registra uma candidata no MLflow
make app      # abre a interface em http://localhost:8501
make test     # roda os testes
```

**LLM (opcional).** Sem chave, a explicação é montada por regras e o resto funciona igual. Para ligar, copie `.env.example` para `.env` e preencha `ANTHROPIC_API_KEY`. O modelo padrão é `claude-opus-5-5`; troque com `ZD_LLM_MODEL`.

## O que já funciona

| Parte | O que faz |
|---|---|
| Pipeline de dados | API do Wikilivros → wikitexto bruto (`data/raw`) → receitas normalizadas (`data/processed`) + manifesto com hashes, estatísticas e licença |
| Normalização de ingredientes | "3 xícaras (chá) de açúcar" → `açúcar`, 3, xícara; plural, acento, sinônimos (maisena → amido de milho) |
| Agente Despensa | validade e urgência por regras fixas; itens vencidos nunca entram nas sugestões |
| Agente Restrições | alergias e restrições (palavra solta ou grupo: lactose, glúten, frutos do mar, carne); erra para o lado seguro |
| Agente Receitas | recuperação híbrida (embeddings multilíngues no ChromaDB + índice invertido de ingredientes) e ranking determinístico |
| Agente Explicação | LLM redige a justificativa citando `[R1]`, `[R2]`…; se falhar, recusar ou citar receita inexistente, cai para regras |
| Interface | despensa, sugestões com fonte e licença, rastro dos agentes, comparação LLM × RAG, painel de dados e métricas |

Ranking: `pontuação = 0,5 × cobertura + 0,4 × urgência aproveitada + 0,1 × similaridade` (com "vontade" preenchida: 0,4 · 0,3 · 0,3).

## Dados

- **Fonte:** [Wikilivros — Livro de receitas](https://pt.wikibooks.org/wiki/Livro_de_receitas), licença [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.pt). Cada sugestão mostra o link da página e a data da revisão.
- **Números da base atual:** 2.058 páginas → 2.088 receitas (412 de bebidas alcoólicas ficam fora por decisão do projeto; 5 duplicatas removidas). 1,9% das linhas de ingrediente ficam sem ingrediente reconhecido.
- **Conferência manual:** 93 de 100 linhas sorteadas normalizadas corretamente; os erros encontrados viraram casos de teste.

## Métricas medidas

Teste "known item" (300 receitas sorteadas, semente 42): a despensa recebe 70% dos ingredientes de uma receita e medimos se ela volta entre as k primeiras.

| Modo | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| só semântica (embeddings) | 0,06 | 0,13 | 0,19 | 0,09 |
| só ingredientes (índice invertido) | 0,71 | 0,85 | 0,88 | 0,77 |
| híbrida + ranking (usada no app) | 0,33 | 0,72 | 0,81 | 0,49 |

Leitura: embeddings sozinhos não bastam para "lista de ingredientes → receita". O ranking do app perde para a busca por ingredientes nesse teste porque ele favorece receitas que a pessoa consegue fazer **agora** (maior cobertura, uso do que vence) e não a receita "certa" do sorteio. Precisamos de uma métrica alinhada ao objetivo (cobertura e itens urgentes aproveitados, avaliados com usuários) — ponto a discutir com o professor.

## MLOps (registro, portão e volta atrás)

Decisão em [ADR 0007](docs/adr/0007-mlops-registro-portao-e-retorno.md); desenho em `docs/superpowers/specs/2026-11-19-mlops-design.md`. O MLflow guarda cada versão candidata (parâmetros, métricas, configuração, commit e conjunto de avaliação). Uma versão só vira a **produção** (marca `producao`) se passar pelo portão: atingir os mínimos e não piorar além da tolerância contra a produção atual, no mesmo conjunto de avaliação. Os limites ficam em `mlops/gates.yaml` (mudar exige ADR).

| Modelo registrado | O que é | Métricas do portão | Estado |
|---|---|---|---|
| `zd-rag-receitas` | configuração da recuperação (pesos do ranking, modelo de embeddings) | `hybrid_recall_at_10` ≥ 0,80 e `hybrid_mrr` ≥ 0,45; tolerância 0,02 | ligado: `make eval` registra, o app lê |
| `zd-frescor` | classificador de frescor (Vitor) | `f1_macro` ≥ 0,85 e `recall_estragado` ≥ 0,85; tolerância 0,01 | só os limites existem |
| `zd-explicacao` | prompt e explicação com LLM | `citacoes_validas` ≥ 0,95 e `afirmacoes_seguro` = 0; tolerância 0,02 | só os limites existem |

```bash
make eval                                              # mede e registra uma candidata (vN)
make status                                            # mostra a versão em produção de cada modelo
make promote  MODEL=zd-rag-receitas VERSION=2 MOTIVO="pesos novos"   # passa pelo portão
make rollback MODEL=zd-rag-receitas VERSION=1 MOTIVO="v2 piorou"     # volta atrás; o motivo é obrigatório
make mlflow-ui                                         # interface do MLflow em http://localhost:5000
```

O CLI sai com código 1 quando o portão recusa (e mostra as razões); o make mostra o erro e sai com status diferente de zero. `make rollback` não passa pelo portão. Promover a versão que já está em produção, ou voltar para ela, não faz nada e não gera evento no log. Se `promote` ou `rollback` falharem por versão ou modelo inexistente, a marca não muda.

**Exemplo de fluxo.** Para testar pesos novos sem editar código, grave um JSON com a configuração (`RagConfig`) e avalie com `--config`. O `embedding_model` do JSON precisa ser o mesmo do índice; se for diferente, a avaliação aborta. O `make eval` não repassa argumentos, então use o comando direto:

```bash
.venv/bin/python -m zerodesperdicio.evaluation retrieval --config arq.json --registrar
make status
make promote MODEL=zd-rag-receitas VERSION=2 MOTIVO="pesos novos"   # APROVADO ou RECUSADO, com as razões
make rollback MODEL=zd-rag-receitas VERSION=1 MOTIVO="queda de qualidade"
```

**Atenção ao `$` no MOTIVO.** O make expande `$` nos valores da linha de comando: escreva `$$` para um `$` literal (ex.: `MOTIVO="custo de $$5"`). Aspas e crases não são interpretadas pelo shell.

**Onde ficam as coisas**
- `mlruns/` (banco SQLite `mlflow.db` e artefatos): local e ignorado no git, sempre na raiz do projeto, de qualquer pasta que o comando rode.
- `reports/promocoes.jsonl`: uma linha por promoção, recusa ou volta atrás (quando, modelo, ação, de/para, quem, motivo, razões, métricas). Não vai para o git (está no `.gitignore`): o log é local por máquina enquanto o armazenamento compartilhado estiver em aberto.
- `MLFLOW_TRACKING_URI`: aponta para outro servidor do MLflow em vez do SQLite local (`make mlflow-ui` abre só o banco local).

**No app.** O app carrega a configuração do `zd-rag-receitas` marcada como `producao` e mostra a versão na aba "Dados e métricas". Se o registro estiver vazio, sem a marca, fora do ar ou incompatível, o app **não para**: usa a configuração padrão do código e mostra um aviso. Promoções e voltas atrás só valem depois de reiniciar o app.

## Estrutura

```
src/zerodesperdicio/   pipeline.py · wikibooks.py · ingredients.py · ingredient_rules.py · pantry.py
                       ranking.py · index.py · llm.py · explain.py · agents.py · evaluation.py
                       rag_config.py · app_support.py
                       mlops/ (gates.py · registry.py · rag.py · __main__.py)
app/streamlit_app.py   interface
mlops/gates.yaml       limites do portão de promoção
tests/                 testes (pytest), sem rede
docs/adr/              decisões de arquitetura
docs/superpowers/plans plano de implementação deste MVP
```

## Próximos passos

- Semana 4: agente de Visão (foto → alimentos com classificador leve; foto → data de validade por OCR, lida em voz alta).
- Semana 5: ligar `zd-frescor` e `zd-explicacao` ao registro (hoje só o RAG está ligado), DVC nos dados e no pipeline, CI no GitHub Actions com portão de métricas, Docker, monitoramento e retreino com as correções dos usuários.
- Métrica de utilidade com usuários reais e meta numérica para cada métrica.
