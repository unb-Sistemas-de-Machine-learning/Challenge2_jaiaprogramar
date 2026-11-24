# Semana 4 — dados de frescor, licença e DVC

Este plano cobre a preparação da base de imagens na Semana 4, deixando a saída pronta para embeddings CLIP em ONNX, comparação zero-shot versus regressão logística e rastreamento futuro no MLflow. O repositório atual já ignora `data/raw/` e `data/processed/`; por isso, a proposta usa `datasets/freshness/` para que os dados de visão possam ser versionados pelo DVC sem interferir no pipeline de receitas.

## Plano prático em dupla

| Etapa | Eduardo | Pedro | Resultado verificável |
|---|---|---|---|
| 1. Fonte e licença | Escolhe a base candidata e registra URL, versão/data, autor e licença declarada; confere os termos na página do Kaggle e na fonte original das imagens. | Faz revisão independente dos termos e sinaliza restrições de redistribuição, uso comercial, atribuição e derivados. | `docs/datasets/freshness-sources.md` preenchido; decisão explícita: aprovado, aprovado com condições ou bloqueado. |
| 2. Recebimento | Baixa para `datasets/freshness/raw/<source_id>/` (sem editar o original); guarda instruções/versão e, quando possível, checksum do arquivo de origem. | Confere que nenhuma credencial, cookie, dado pessoal ou material sem licença compatível foi adicionado ao Git. | Fonte bruta preservada e reobtível; registro de aquisição/licença no repositório. |
| 3. Inspeção e normalização | Executa o script abaixo e revisa exemplos/contagens por classe e produto. | Audita amostra das imagens e o manifesto; confirma rótulos e duplicatas; registra exclusões. | `organized/` com `produto_fresco/`, `produto_estragado/` e `manifest.csv`. |
| 4. DVC e colaboração | Inicializa DVC (se ainda não existe), adiciona a base aprovada e configura remote escolhido pela equipe. | Reproduz `dvc pull` em outro clone/ambiente e valida hash/manifesto antes de aprovar o PR. | `.dvc` e metadados de configuração commitados; conteúdo grande no remote DVC; procedimento reproduzido. |
| 5. Preparação para ML | Mantém fonte, produto e hash em cada linha; entrega rótulos visuais operacionais. | Define com a equipe splits e avaliação sem vazamento, e registra decisões para pipeline CLIP/MLflow. | Manifesto e divisão reproduzível; sem tratar rótulo visual como diagnóstico de segurança alimentar. |

### Critérios de conclusão

- Todas as imagens aceitas têm classe conhecida, produto (ou `produto_nao_informado`), fonte, caminho relativo e SHA-256 no manifesto.
- Imagens corrompidas, duplicadas e não classificadas são contabilizadas; os originais não são apagados nem alterados.
- Licença e atribuição estão registradas antes de publicar ou compartilhar dados/artefatos.
- Pedro consegue obter a mesma versão com `dvc pull`; arquivos grandes não são adicionados diretamente ao Git.
- A divisão de avaliação é feita por grupo/fonte/produto quando apropriado (e não por cópias quase idênticas aleatoriamente), para reduzir vazamento entre treino e teste.

## Checklist de licença e proveniência

Antes de usar cada dataset ou conjunto de fotos, registrar e conferir:

1. **Identidade da fonte:** nome oficial, URL da página do Kaggle e URL da fonte original indicada pelo publicador; autor/organização; versão ou data de captura; data do download; pessoa responsável.
2. **Licença efetiva:** texto/link da licença exibida para aquela versão. Não inferir licença de um campo ausente, do fato de o dataset ser público, nem dos termos gerais do Kaggle. Se não houver licença explícita ou ela for ambígua, não distribuir nem usar em produto até esclarecer com o titular.
3. **Permissões e obrigações:** uso acadêmico/comercial, cópia e redistribuição, atribuição/citação, compartilhamento de adaptações, restrições de uso, prazo/território e exigências de aviso. Verificar também termos de cada fonte externa embutida: imagens podem ter direitos diferentes dos metadados ou do pacote.
4. **Compatibilidade:** comparar a licença com a finalidade do projeto, o acesso esperado ao repositório e o armazenamento remoto DVC. Se os termos não permitirem redistribuição, não enviar conteúdo a um remote acessível aos colaboradores sem autorização; avaliar manter a base fora do remote e documentar como obtê-la legalmente.
5. **Evidência:** guardar em `docs/datasets/freshness-sources.md` URL, data, versão, titular, licença/link e trecho/arquivo de termos consultado (quando permitido), atribuição exigida, decisão e justificativa. Não copiar grandes textos protegidos; registrar link e resumo. Manter citações/créditos requeridos junto ao conjunto e em entregáveis derivados.
6. **Revisão:** Eduardo preenche; Pedro confere. Atualizar registro se a versão, fonte, licença ou finalidade mudar. Uma aprovação técnica não substitui análise jurídica quando houver dúvida material.

### Fotos próprias e fotos de terceiros

Antes de misturar fotos próprias, obter consentimento/autorização documentada de quem fotografou e de quem detém os direitos, quando não for a mesma pessoa; definir finalidade, prazo, acesso, possibilidade de retirada e se a imagem poderá ser usada para treinamento ou demonstração. Evitar pessoas, rótulos com dados pessoais, telas, documentos, localização/EXIF e conteúdo identificável; remover metadados desnecessários. Fotos de terceiros exigem permissão/licença compatível, não apenas crédito. Não reutilizar dados pessoais ou imagens obtidas em contexto doméstico sem base e autorização adequadas. Manter `source` distinto por origem e não misturar conjuntos até validar compatibilidade de licença e rotulagem.

## Estrutura recomendada

```text
datasets/
└── freshness/
    ├── raw/                         # aquisição preservada por fonte/versão; nunca editar
    │   ├── kaggle_fvd_<versao>/
    │   └── own_photos_<versao>/
    ├── organized/                   # saída do script; candidata a rastreamento DVC
    │   ├── produto_fresco/
    │   │   └── <produto>/
    │   ├── produto_estragado/
    │   │   └── <produto>/
    │   └── manifest.csv
    ├── metadata/                    # recibos, descrição de split e checksums não sensíveis
    │   └── README.md
    └── .gitignore                   # opcional, regras geradas pelo DVC

docs/datasets/
└── freshness-sources.md             # inventário de fontes/licenças, não dados de imagem
```

`raw/` pode permanecer fora do DVC se a licença não autorizar cópia/redistribuição ou se o arquivo for recuperável por URL e checksum; nesse caso, DVC rastreia apenas `organized/` se isso for permitido, ou a equipe mantém os dados fora do remote e documenta o procedimento. Não commitar imagens ou arquivos de licença restritivos no Git. O DVC não concede direitos de uso: a autorização precisa vir da fonte.

O manifesto gerado inclui `image_path`, `label`, `source`, `product`, além de `status`, `sha256` e `notes` para auditoria. Os caminhos são relativos a `organized/`. Linhas com `status` diferente de `accepted` são registros de auditoria e não devem entrar no treinamento.

## Organizar e validar imagens

O script requer Python 3.10+ e Pillow:

```bash
python -m pip install Pillow
python scripts/prepare_freshness_dataset.py \
  --input-dir datasets/freshness/raw/kaggle_fvd_<versao> \
  --output-dir datasets/freshness/organized \
  --source kaggle_fvd_<versao>
```

Layout de entrada aceito pelo script (aliases comuns em inglês/português são reconhecidos; confira a lista em `LABEL_ALIASES`):

```text
<entrada>/<produto>/fresh/<imagem.jpg>
<entrada>/<produto>/spoiled/<imagem.jpg>
# ou
<entrada>/produto_fresco/<imagem.jpg>
<entrada>/produto_estragado/<imagem.jpg>
```

O script verifica extensão e decodificação da imagem, identifica classe/produto pelos diretórios, detecta duplicatas binárias por SHA-256 e copia apenas itens únicos válidos. Ele não altera nem apaga a entrada. Casos sem diretório de classe reconhecido são ignorados e contabilizados no resumo (não aparecem como imagens de treino no CSV); examine a saída e corrija o mapeamento quando necessário. Inspecione manualmente amostras por classe, pois integridade de arquivo não garante rótulo correto.

O rótulo é apenas uma categoria visual operacional — **“fresco” versus “começando a estragar/estragado”** — e não deve ser apresentado como “seguro”, diagnóstico de segurança alimentar ou garantia de consumo. No produto, seguir a regra de linguagem do sistema: nunca declarar um alimento seguro a partir de uma imagem.

## DVC: inicialização e colaboração

Execute na raiz do repositório. Se o projeto ainda não for um clone Git, primeiro use o fluxo de Git já adotado pela equipe (não inicialize uma história separada se houver remoto existente). DVC é uma dependência de desenvolvimento; pode ser instalado no ambiente virtual ou via `pipx`:

```bash
python -m pip install 'dvc>=3,<4'
# se ainda não houver .dvc/:
dvc init

dvc add datasets/freshness/organized
# revisar o .gitignore gerado dentro de organized/ e o arquivo datasets/freshness/organized.dvc
git status --short
```

O conteúdo pesado fica no cache/remote do DVC; o Git deve receber o arquivo `.dvc`, as regras `.gitignore` criadas pelo DVC, os scripts e a documentação. Confira `git check-ignore -v datasets/freshness/organized/manifest.csv` e `git status`; não force arquivos grandes para o Git. `dvc add` pode gerar um `.gitignore` interno que exclui o conteúdo rastreado.

### Configurar um remote (escolher com a equipe)

Opções usuais: armazenamento de objeto compatível com S3, Google Cloud Storage, Azure Blob ou SSH/servidor acessível à equipe. Custos, credenciais, política de acesso, região e retenção dependem do serviço e não são definidos aqui. Use um bucket/prefixo privado com acesso mínimo necessário. Não colocar segredo no Git.

Exemplo genérico S3 (substitua valores; credenciais devem vir do perfil/variáveis seguros do ambiente):

```bash
dvc remote add -d freshness s3://<bucket>/<prefixo>/freshness
# se for apropriado, compartilhar somente a configuração não secreta:
dvc config core.autostage true   # opcional
# credenciais fora do Git, por exemplo via perfil AWS configurado no ambiente
```

Para configuração local/secreta de DVC, use `dvc remote modify --local freshness ...`; isso grava em `.dvc/config.local`, que não deve ser commitado. Para qualquer provedor, consultar a documentação do DVC e usar a integração/credencial apropriada. `dvc push` e `dvc pull` pressupõem remote configurado e permissão de acesso.

### Fluxo sem conflitos para Eduardo e Pedro

1. **Eduardo (responsável pela atualização):** trabalhar em branch; preparar dataset e manifesto; `dvc add datasets/freshness/organized`; executar validações; `dvc push`; fazer commit dos arquivos pequenos (por exemplo `.dvc`, `.gitignore`, script/documentação) e abrir PR. Não enviar as imagens ao Git.
2. **Pedro (revisor):** atualizar branch/PR; instalar DVC e dependências; executar `dvc pull`; verificar que a versão e manifesto correspondem ao `.dvc`; revisar fonte, licença e amostra de imagens. Aprovar somente após confirmar acesso reproduzível e ausência de dados indevidos.
3. **Nova versão:** não sobrescrever em paralelo. Um responsável por vez atualiza a pasta rastreada; registrar versão da origem e repetir `dvc add`/`dvc push`/commit. DVC detecta alteração de conteúdo e o arquivo `.dvc` passa a apontar para novo hash. Em caso de conflito, decidir a versão por PR e refazer `dvc pull`/add — não editar hashes manualmente.
4. **Clone novo:** `git clone ...`, instalar Git/DVC, configurar credencial/remote, depois `dvc pull`. Um colaborador sem autorização de licença ou de acesso ao remote não deve receber os dados.

Comandos de sincronização após remote e arquivos `.dvc` já existentes:

```bash
git pull --rebase
dvc pull
# após alteração autorizada
dvc add datasets/freshness/organized
dvc push
git add datasets/freshness/organized.dvc datasets/freshness/organized/.gitignore docs scripts
git commit -m "Add freshness image dataset version"
git push
```

Ajuste os caminhos do `git add` conforme o `.gitignore` que o DVC criar e os arquivos realmente modificados. Se os dois colaboradores alterarem a base ao mesmo tempo, não fazer merge de conteúdo manualmente: escolher uma atualização por vez e reexecutar a etapa DVC.

## Integração futura: CLIP/ONNX e MLflow

Trate `manifest.csv` como catálogo de entradas e filtre `status == accepted`. Faça splits determinísticos e registre semente/critério; mantenha imagens de mesma origem, duplicatas perceptuais ou sessões de captura no mesmo split quando necessário. A comparação obrigatória deve usar o mesmo conjunto de teste: (1) CLIP zero-shot com prompts previamente documentados e (2) classificador leve (por exemplo, regressão logística) treinado sobre embeddings CLIP extraídos em ONNX/CPU. Registrar métricas por classe, matriz de confusão, versão/hash do dataset, versão do modelo ONNX, prompts, configuração, split e métricas no MLflow. Aprovação de modelo deve depender de suíte de testes sem regressão, não apenas de acurácia agregada. Isso prepara a próxima etapa, mas não implementa nem substitui o pipeline de modelo.
