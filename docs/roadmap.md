# Roteiro do projeto

Entrega final: **07/12/2026** (aplicação, relatório e apresentação). Este arquivo é atualizado a cada etapa concluída.

## Equipe e papéis

Papéis segundo Kreuzberger, Kühl e Hirschl (2023). Em equipe pequena, cada pessoa acumula papéis.

| Pessoa | Papel | Foco |
|---|---|---|
| Ricardo | Arquiteto (R2) + engenharia de software (R5) | arquitetura, agentes, integração do app |
| Pedro | Engenheiro de ML / MLOps (R6, R7), com frente de LLMOps | MLflow, registro de versões, portões, DVC, CI, monitoramento |
| Eduardo | Engenheiro de dados (R4) | ingestão e versionamento de imagens e documentos |
| Vitor | Cientista de dados (R3) — visão | classificador de frescor, leitura de validade |
| Gabriel | Cientista de dados (R3) — RAG e LLM | base de documentos, roteador, avaliação de fidelidade |

## Concluído

- [x] **v0.1 (11/11):** pipeline de dados do Wikilivros (2.088 receitas), normalização de ingredientes, despensa com validade, recuperação híbrida, ranking determinístico, agentes Despensa/Restrições/Receitas/Explicação, interface Streamlit, avaliação de recuperação, ADRs 0001–0003.

## Semana 4 — até 22/11: câmera e visão

- [ ] Diagrama da arquitetura atualizado (Ricardo)
- [ ] Tela da câmera com dois modos e agente de Visão (Ricardo) — ADR 0004
- [ ] Base de frescor baixada, licença confirmada, versionada no DVC (Eduardo)
- [ ] Linha de base zero-shot e primeiro classificador treinado (Vitor) — ADR 0005
- [ ] PDFs divididos em trechos com página e 30 perguntas de avaliação (Gabriel) — ADR 0006
- [~] Parcial: registro, marca `producao`, portão e volta atrás feitos (zd-rag-receitas); DVC no pipeline pendente (Pedro) — ADR 0007

## Semana 5 — 23 a 29/11: integração completa

- [ ] Alerta de frescor aumenta a prioridade; data lida pela câmera preenche o item (Ricardo)
- [ ] Ingestão dos PDFs com manifesto; mutirão de fotos rotuladas (Eduardo + todos)
- [ ] Classificador melhorado e testado nas fotos do grupo; comparação de motores de OCR (Vitor)
- [ ] Roteador entre as duas bases e prompt com citação de página (Gabriel)
- [ ] CI com portão de métricas; LLMOps (versão do prompt, log de chamadas); monitoramento e gatilho de retreino; Docker (Pedro)

## Semana 6 — 30/11 a 06/12: testes e relatório

- [ ] Teste com usuários (2 pessoas por integrante)
- [ ] Avaliação final e congelamento das versões de produção (Pedro)
- [ ] Relatório (cada pessoa escreve a seção da sua parte) e ensaio da apresentação

## Combinados

- Uma branch por tarefa e Pull Request revisado por uma pessoa antes de entrar na `main`.
- Decisão importante vira ADR em `docs/adr`.
- Chaves de API ficam no `.env`, nunca no git. Dados e modelos grandes vão pelo DVC.
