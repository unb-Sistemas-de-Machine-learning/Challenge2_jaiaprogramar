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

Comandos: `make promote MODEL=<nome> VERSION=<n>` (roda o portão e só move a marca se aprovar) e `make rollback MODEL=<nome> VERSION=<n>` (move a marca e registra o motivo). Cada promoção e retorno fica nas tags do registro e em `reports/promocoes.jsonl`.

### Dados e pipeline

- DVC versiona `data/raw`, `data/processed`, `data/eval` e os índices.
- `dvc.yaml` encadeia: receitas (`fetch → build → index → eval`) e frescor (`preparar → treinar → avaliar`). Cada etapa registra no MLflow.
- Conjuntos de avaliação ficam em `data/eval/` com hash no manifesto. Um teste garante que nenhum arquivo de avaliação aparece no treino.
- Contrato com o Eduardo: imagens em `data/raw/frescor/<produto>__<fresco|estragado>/*.jpg`.
- Contrato com o Vitor: o treino fica em `src/zerodesperdicio/freshness/train.py`, recebe vetores e rótulos e devolve o modelo. O MLOps entrega uma linha de base funcionando (regressão logística) para o pipeline rodar de ponta a ponta antes das melhorias do Vitor *(minha decisão)*.

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
