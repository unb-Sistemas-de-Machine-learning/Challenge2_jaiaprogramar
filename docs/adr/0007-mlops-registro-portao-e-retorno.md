# ADR 0007 — MLOps: registro de versões, portão de promoção e volta atrás

**Status:** Proposto · 19/11/2026 · desenho completo em `docs/superpowers/specs/2026-11-19-mlops-design.md`

## Contexto
O sistema tem um modelo treinado (classificador de frescor) e componentes com LLM (RAG e explicação). Precisamos saber se cada versão nova melhora ou piora e voltar rapidamente quando uma versão nova errar mais ou alucinar.

## Decisão
- **MLflow** registra cada treino e cada avaliação: versão dos dados, parâmetros, commit do código, métricas e artefatos.
- **Registro de modelos** com a marca (alias) `producao`. O app carrega sempre a versão marcada.
- **Portão de promoção:** uma versão candidata só recebe a marca se atingir os mínimos e não piorar além da tolerância em relação à versão em produção, no mesmo conjunto fixo de avaliação. Limites em arquivo versionado.
- **Volta atrás** = devolver a marca `producao` à versão anterior, com um comando; cada promoção e retorno fica registrado.
- O mesmo vale para a configuração do RAG e o prompt do LLM (LLMOps).
- **DVC** versiona os dados e encadeia o pipeline; **CI** roda testes e o portão a cada PR.

## Consequências
- Toda versão em produção é reproduzível (dados + código + parâmetros).
- Conjuntos de avaliação fixos precisam ser mantidos e nunca usados no treino.
- Onde ficam o servidor do MLflow e o armazenamento do DVC (compartilhado ou local) ainda está em aberto.
