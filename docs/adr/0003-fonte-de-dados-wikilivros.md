# ADR 0003 — Fonte de receitas: Wikilivros (CC BY-SA 4.0)

**Status:** Proposto · 11/11/2026

## Contexto
O desafio pede evidências confiáveis. Bases grandes (RecipeNLG, Food.com) estão em inglês e exigem cadastro; não achamos base aberta de receitas brasileiras em JSON com licença clara.

## Decisão
Usar o "Livro de receitas" do Wikilivros em português: ~2.000 páginas, licença CC BY-SA 4.0, histórico de revisões e API oficial.
- Coleta pela API MediaWiki, no limite de 10 requisições/min, respeitando `Retry-After`.
- Cada receita guarda URL, id e data da revisão e licença; a interface mostra a atribuição.
- Bebidas alcoólicas ficam fora da base.

## Consequências
- Receitas brasileiras reais e rastreáveis; qualidade variável (wiki), revisões antigas (muitas de 2011).
- Formatação irregular exige parser próprio (dois layouts principais) — coberto por testes e por conferência manual (93/100).
- Obras derivadas da base precisam manter a mesma licença.
