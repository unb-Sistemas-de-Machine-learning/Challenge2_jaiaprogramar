# ADR 0006 — RAG com duas bases e agente roteador

**Status:** Aceito · 19/11/2026

## Contexto
O RAG atual recupera só receitas. Perguntas como "o tomate está ficando mole, o que faço?" pedem orientação confiável (conservação, aproveitamento integral), não apenas receitas.

## Decisão
- **Base 1 — receitas** (Wikilivros, já existente).
- **Base 2 — documentos:** Guia Alimentar para a População Brasileira e cartilhas de aproveitamento integral e conservação, divididos em trechos com o número da página.
- **Agente roteador** decide em qual base buscar (ou nas duas) conforme a pergunta e o estado da despensa.
- Quando o classificador de frescor marca um item como "começando a estragar", o sistema busca na base 2 o que fazer com ele e na base 1 receitas que o usem.
- As respostas citam a fonte e a página; citação inexistente invalida a resposta (mesma regra do ADR 0002).

## Consequências
- Avaliação própria para a base 2: perguntas com a página esperada (a página certa foi recuperada?) e fidelidade (a resposta só afirma o que está nos trechos?).
- Cada documento entra com fonte, licença e data no manifesto.
- Mais um ponto de alucinação a vigiar no LLMOps (ADR 0007).
