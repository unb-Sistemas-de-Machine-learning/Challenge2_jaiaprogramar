# Registro de decisões (ADRs)

Cada decisão de arquitetura, dados ou modelo vira um arquivo curto aqui: contexto, decisão e consequências. Status possíveis: **Proposto** (ainda em discussão), **Aceito**, **Substituído por ADR N**.

| # | Decisão | Status | Data |
|---|---|---|---|
| [0001](0001-recuperacao-hibrida-sem-pytorch.md) | Recuperação híbrida com embeddings em ONNX (sem PyTorch) | Proposto | 11/11/2026 |
| [0002](0002-regras-deterministicas-llm-so-explica.md) | Regras determinísticas decidem; o LLM só explica | Proposto | 11/11/2026 |
| [0003](0003-fonte-de-dados-wikilivros.md) | Fonte de receitas: Wikilivros (CC BY-SA 4.0) | Proposto | 11/11/2026 |
| [0004](0004-camera-com-dois-usos.md) | Câmera com dois usos: validade de industrializados e frescor de hortifrúti | Aceito | 19/11/2026 |
| [0005](0005-classificador-de-frescor.md) | Classificador de frescor: embeddings do CLIP + modelo leve | Proposto | 19/11/2026 |
| [0006](0006-rag-com-duas-bases-e-roteamento.md) | RAG com duas bases e agente roteador | Aceito | 19/11/2026 |
| [0007](0007-mlops-registro-portao-e-retorno.md) | MLOps: registro de versões, portão de promoção e volta atrás | Proposto | 19/11/2026 |

**Como escrever um ADR:** copie um existente, numere em sequência, preencha em até uma página e abra um PR. Mudou de ideia? Não apague: crie um novo ADR e marque o antigo como "Substituído por".
