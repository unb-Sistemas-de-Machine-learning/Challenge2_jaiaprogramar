# ADR 0005 — Classificador de frescor: embeddings do CLIP + modelo leve

**Status:** Proposto · 19/11/2026

## Contexto
Reconhecer qualquer alimento numa foto é um problema amplo e caro. A pergunta útil para o app é mais estreita: este item de hortifrúti parece fresco ou começando a estragar? A máquina de desenvolvimento não tem GPU e o projeto evita PyTorch (ADR 0001).

## Decisão
- Extrair um vetor de cada foto com o CLIP ViT-B/32 em ONNX (`Qdrant/clip-ViT-B-32-vision`, via fastembed, ~340 MB, CPU).
- Treinar um classificador leve (regressão logística, scikit-learn) sobre esses vetores.
- Linha de base obrigatória: o mesmo CLIP sem treino (zero-shot), comparando a foto com textos como "foto de tomate estragado".
- Dados: base pública "saudável × podre" de frutas e hortaliças (candidata: *Fruit and Vegetable Diseases*, Kaggle, 14 produtos; licença a confirmar) somada a fotos tiradas pelo grupo.

## Consequências
- Treino em segundos na CPU: dá para registrar muitas versões e comparar (ADR 0007).
- Fotos de banco de imagens diferem de fotos de cozinha; por isso as fotos do grupo entram no conjunto de teste fixo.
- Métricas: F1 por classe e matriz de confusão. Errar "estragado" como "fresco" pesa mais e é acompanhado à parte.
