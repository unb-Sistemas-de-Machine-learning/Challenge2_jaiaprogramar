# ADR 0002 — Regras determinísticas decidem; o LLM só explica

**Status:** Proposto · 11/11/2026

## Contexto
Validade, alergias e "o que dá para fazer" têm consequência real. LLMs podem inventar ingredientes ou receitas.

## Decisão
- Validade, prioridade, restrições e ranking são **código determinístico e testado**.
- O LLM (Claude, SDK oficial, `claude-opus-5-5`, esforço baixo, fallback do servidor ligado) recebe só o contexto recuperado e redige a explicação citando `[R#]`.
- A resposta é validada: sem citação, citação inexistente, recusa ou erro de rede → explicação por regras, com aviso na tela.
- O app funciona sem chave de API.

## Consequências
- Comportamento reproduzível e auditável; testes não dependem de rede.
- O LLM agrega clareza, não decisão. O texto das receitas é tratado como dado (proteção contra instruções embutidas).
