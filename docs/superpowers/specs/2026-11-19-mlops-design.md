# Desenho — MLOps e LLMOps do ZeroDesperdício

**Responsável:** Pedro · **Data:** 19/11/2026 · **Decisão registrada em:** ADR 0007

## Por quê

O ZeroDesperdício terá um modelo treinado (classificador de frescor, ADR 0005) e partes com LLM (RAG com duas bases e explicação, ADRs 0002 e 0006). Cada versão nova pode melhorar ou piorar, e uma versão com LLM pode passar a alucinar. Precisamos:

1. saber, com números, se cada versão nova evoluiu ou regrediu;
2. reproduzir qualquer versão (dados + código + parâmetros);
3. voltar para a versão anterior com um comando;
4. perceber em uso quando a qualidade cai, e usar as correções das pessoas para treinar de novo.

## Objetivos

- **Rastreio:** todo treino e toda avaliação registrados no MLflow.
- **Registro de versões:** modelos registrados, com a marca `producao` apontando para a versão que o app usa.
- **Portão:** uma versão só vira `producao` se passar nos mínimos e não piorar além da tolerância em relação à atual, no mesmo conjunto fixo de avaliação.
- **Volta atrás:** devolver a marca para uma versão anterior, com registro de quem, quando e por quê.
- **Dados versionados e pipeline reproduzível** com DVC.
- **CI:** testes e checagem do portão a cada Pull Request.
- **LLMOps:** prompt versionado, cada chamada ao LLM registrada (modelo, versão do prompt, tokens, tempo, custo estimado, recusa, citação inválida), avaliação de fidelidade.
- **Monitoramento e retreino:** painel no app; correções das pessoas viram dados rotulados; gatilho de retreino.

## Não objetivos

- Feature store, Kubernetes, servidor de inferência separado: o tamanho do projeto não justifica (o próprio modelo de referência pede justificar cada componente).
- Treino em GPU ou PyTorch.
- Teste A/B com usuários reais em produção.

## Anti-objetivos

- Promover uma versão sem compará-la com a anterior no mesmo conjunto de avaliação.
- Avaliar com dados que foram usados no treino.
- Depender de chave de API para rodar os testes do CI.
- Subir dados, modelos ou chaves no git.

## Restrições

- CPU, ~5 GB de RAM, sem GPU. Tudo em ONNX ou scikit-learn.
- Prazo: infraestrutura básica na semana 4 (até 22/11), CI, LLMOps e monitoramento na semana 5 (até 29/11).
- O classificador é treinado pelo Vitor; os dados chegam pelo Eduardo; o RAG de documentos é do Gabriel. O MLOps define os contratos e envolve o trabalho deles, sem substituí-lo.

## Como funciona

### Modelos registrados

| Nome no registro | O que é | Avaliado em |
|---|---|---|
| `zd-frescor` | vetor CLIP (ONNX) → regressão logística | conjunto de teste fixo de fotos (base pública + fotos do grupo) |
| `zd-rag-receitas` | configuração da recuperação: modelo de embeddings, pesos do ranking, nº de candidatas, hash da base e do índice | teste "known item" fixo (300 receitas, semente 42) |
| `zd-explicacao` | prompt versionado + modelo do LLM | 30 despensas fixas; citações válidas, sem afirmação de segurança alimentar |

Depois: `zd-rag-documentos` (Gabriel) e `zd-leitura-validade` (Vitor) entram com o mesmo padrão.

O app carrega cada componente pela marca `producao`. Se o registro estiver vazio ou fora do ar, o app usa a configuração padrão do código e mostra um aviso *(minha decisão: o app nunca para por causa do MLflow)*.

### Portão de promoção

Limites em `mlops/gates.yaml` (versionado). Valores iniciais *(minha decisão: partem do que já foi medido e mudam por ADR)*:

| Modelo | Mínimo absoluto | Tolerância contra `producao` |
|---|---|---|
| `zd-frescor` | F1 macro ≥ 0,85 e recall de "estragado" ≥ 0,85 | não cair mais de 0,01 |
| `zd-rag-receitas` | recall@10 ≥ 0,80 e MRR ≥ 0,45 | não cair mais de 0,02 |
| `zd-explicacao` | citações válidas ≥ 95%; afirmações de "seguro" = 0 | não cair mais de 2 pontos |

Comandos: `make promote MODEL=<nome> VERSION=<n> MOTIVO=<texto>` (roda o portão e só move a marca se aprovar; `MOTIVO` opcional), `make rollback MODEL=<nome> VERSION=<n> MOTIVO=<texto>` (move a marca; `MOTIVO` obrigatório) e `make status` (versão em produção de cada modelo). Promover ou voltar para a versão que já está em produção não faz nada e não gera evento no log. Cada promoção e retorno fica nas tags do registro e em `reports/promocoes.jsonl`.

### Dados e pipeline

- DVC versiona `data/raw`, `data/processed`, `data/eval` e os índices.
- `dvc.yaml` encadeia: receitas (`fetch → build → index → eval`) e frescor (`preparar → treinar → avaliar`). Cada etapa registra no MLflow.
- Conjuntos de avaliação ficam em `data/eval/` com hash no manifesto. Um teste garante que nenhum arquivo de avaliação aparece no treino.
- Contrato com o Eduardo *(decisão de 26/11: adaptar aos caminhos já usados por Eduardo e Vitor)*: `datasets/freshness/organized/{produto_fresco,produto_estragado}/<produto>/` + `manifest.csv`; entram só as linhas com `status == accepted`.
- Contrato com o Vitor *(decisão de 26/11: adaptar aos caminhos já usados por Eduardo e Vitor)*: o treino continua em `zerodesperdicio.vision.train_classifier`. O registro passa a ser `registry.log_candidate("zd-frescor", ..., eval_set=<sha256 do manifesto do conjunto de teste>)` com as métricas `f1_macro` e `recall_estragado`, no lugar de `vision/mlflow_tracking.py` (o portão próprio de 0,80 sai). Imagens e `.joblib` saem do git e vão para o DVC.
- Métricas do portão (nomes reais, sem `@`, em `mlops/gates.yaml`): `zd-rag-receitas` usa `hybrid_recall_at_10` e `hybrid_mrr`; `zd-frescor` usa `f1_macro` e `recall_estragado`; `zd-explicacao` usa `citacoes_validas` e `afirmacoes_seguro` (máximo 0).
- Log de promoções: `reports/promocoes.jsonl`, uma linha JSON por evento, com as chaves `quando` (UTC, ISO 8601), `modelo`, `acao` (`promocao`, `promocao_recusada` ou `retorno`), `de` (versão anterior em produção ou `null`), `para`, `por`, `motivo`, `razoes` (lista) e `metricas` (da versão de destino). O mesmo vai para as tags da versão (`promovida_em`/`promovida_por` ou `retorno_em`/`retorno_por`, e `motivo`).

### CI (GitHub Actions)

- A cada PR: instalar com uv, rodar `pytest` (sem rede e sem chave) e testes do portão com dados mínimos de exemplo.
- Avaliação completa: disparo manual, porque precisa dos dados no armazenamento do DVC.

### LLMOps e monitoramento

- Prompts em arquivos com versão; a versão e o hash vão em cada registro.
- Cada chamada ao LLM é registrada em `data/telemetry/llm_calls.jsonl`.
- Correções das pessoas (data lida errada, frescor errado) vão para `data/feedback/`.
- Aba "Monitoramento" no app: chamadas, tempo p95, % de citações inválidas, recusas, custo estimado, taxa de correção.
- **Gatilho de retreino** *(minha decisão)*: 50 correções novas rotuladas, ou mais de 20% de correção nas últimas 30 previsões. O app mostra o aviso; o retreino roda com `make retrain` e passa pelo portão.

### Docker

`Dockerfile` do app e `docker-compose.yml` com o app e a interface do MLflow.

## Em aberto

- **Onde ficam o MLflow e o armazenamento do DVC:** num serviço compartilhado pelo grupo ou em cada máquina. Isso muda o CI e a configuração. Até a decisão, o MLflow usa um arquivo SQLite local e tudo é trocável por variável de ambiente (`MLFLOW_TRACKING_URI`).

## Fica para o plano

Versões exatas das bibliotecas (MLflow com Python 3.14), base de imagens provisória até a do Eduardo chegar e formato exato dos arquivos de telemetria.
