# ADR 0004 — Câmera com dois usos: validade de industrializados e frescor de hortifrúti

**Status:** Aceito · 19/11/2026

## Contexto
Cadastrar a despensa digitando datas é trabalhoso, e quem tem baixa visão tem dificuldade para ler a validade na embalagem. Frutas e hortaliças não têm data impressa; o que indica urgência é a aparência.

## Decisão
A câmera do app (foto única, sem vídeo ao vivo) terá dois modos:
1. **Produto industrializado:** foto da embalagem → OCR → regras determinísticas acham a data de validade e a separam de fabricação e lote → a pessoa confirma ou corrige antes de salvar.
2. **Fruta ou hortaliça:** foto → classificador "fresco × começando a estragar" → o resultado vira alerta e aumenta a prioridade do item na despensa.

Captura com `st.camera_input` (webcam ou celular) e envio de arquivo como alternativa.

## Consequências
- O app nunca afirma que um alimento está seguro; o classificador só pede conferência e uso rápido.
- Leitura de data errada é um risco: toda leitura é mostrada e confirmada pela pessoa.
- Correções feitas pela pessoa viram dados rotulados para avaliação e retreino (ver ADR 0007).
- O motor de OCR será escolhido por comparação medida (acerto exato da data) nas fotos do grupo.
