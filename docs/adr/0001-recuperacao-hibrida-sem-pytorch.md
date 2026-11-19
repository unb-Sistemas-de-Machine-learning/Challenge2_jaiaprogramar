# ADR 0001 — Recuperação híbrida com embeddings em ONNX (sem PyTorch)

**Status:** Proposto · 11/11/2026

## Contexto
A máquina de desenvolvimento tem ~5 GB de RAM e não tem GPU. A pessoa cadastra uma lista de alimentos em português; as receitas também estão em português.

## Decisão
- Embeddings com `paraphrase-multilingual-MiniLM-L12-v2` via **fastembed** (ONNX, CPU, ~220 MB), guardados no **ChromaDB** (persistente em disco).
- Em paralelo, um **índice invertido** de chaves de ingrediente.
- União das 40 melhores de cada lado e **ranking determinístico** (cobertura, urgência, similaridade).

## Consequências
- Roda em qualquer notebook; nada de PyTorch nesta etapa.
- Medido: embeddings sozinhos têm recall@10 de 0,19 no teste "known item"; a busca por ingredientes, 0,88. Por isso a busca vetorial entra como sinal complementar (vontade/ocasião em texto livre), não como única fonte.
- Alternativas a avaliar no MLflow: `multilingual-e5-large` (2,2 GB), reranker, pesos do ranking.
