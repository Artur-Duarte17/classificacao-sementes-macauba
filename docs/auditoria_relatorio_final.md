# Auditoria técnico-científica do projeto

## Projeto

**Título oficial:** Desenvolvimento de Software Móvel para Identificação de Sementes de Alta Qualidade de Macaúba com Base em Análise Visual e Informações Agronômicas

- **Estudante:** Artur Duarte Monteiro
- **Orientadora:** Rute Quelvia de Faria
- **Instituição:** Instituto Federal Goiano
- **Campus:** Urutaí
- **Período:** 01/08/2025 a 31/07/2026

As informações institucionais foram fornecidas no briefing da auditoria, em
`C:\Users\Artur\.codex\attachments\0518138b-3fcb-42d8-94d8-752f180ec587\pasted-text.txt`.

Esta auditoria foi realizada em 15/06/2026 exclusivamente por leitura de
arquivos, inspeção visual, consulta ao Git e cálculos simples de conferência.
Nenhum modelo foi treinado, nenhum threshold foi alterado, nenhum split foi
recalculado e nenhuma tabela experimental existente foi modificada.

## 1. Identificação da versão auditada

| Item | Valor auditado | Evidência |
|---|---|---|
| Diretório | `C:\Projetos\sementes_ia` | raiz do repositório |
| Branch | `main` | `git branch --show-current` |
| Commit atual | `c3b951a7b74c9aa1d7ffe868a2a00132b691c19d` | `git rev-parse HEAD` |
| Descrição Git | `c3b951a` | `git describe --always --dirty --tags` antes da criação desta auditoria |
| Repositório remoto | `https://github.com/Artur-Duarte17/classificacao-sementes-macauba.git` | `git remote -v` |
| Estado remoto de `main` | mesmo commit `c3b951a7b74c9aa1d7ffe868a2a00132b691c19d` | `git ls-remote --heads origin main` |
| Tag `relatorio-final-v1` | inexistente | `git tag --list` não retornou tags |
| Último commit | 08/06/2026, `Ajusta apresentação do relatório de triagem` | histórico Git |
| Alterações locais anteriores à auditoria | pasta `entrega_final/` não rastreada | `git status --short` |

Os arquivos `entrega_final/Relatorio_Final_Projeto_Sementes.docx` e
`entrega_final/Apresentacao_Final_Projeto_Sementes.pptx` existem localmente,
mas estão ignorados por `*.docx` e não pertencem ao commit auditado. Ambos ainda
contêm campos institucionais genéricos ou incompletos. Portanto, não são fontes
oficiais para esta auditoria.

## 2. Resumo do estado final do projeto

O projeto executou uma pipeline completa de estudo de viabilidade: inventário
dos dados, integração de planilhas e imagens, classificação por diferentes
famílias de modelos, validação por grupo experimental, análise específica do
T6 e tentativa de transformar os scores em triagem preventiva.

O conjunto treinável consolidado contém **703 sementes**, sendo **429
contaminadas** e **274 não contaminadas**. O inventário contém **748 imagens**:
45 não foram associadas a rótulos e 79 registros de rótulo não encontraram
imagem correspondente. As evidências são
`saidas/tabelas/01_inventario/resumo_inventario.csv`,
`saidas/tabelas/03_tabela_mestre/resumo_treinavel.csv`,
`saidas/tabelas/03_tabela_mestre/imagens_sem_rotulo.csv` e
`saidas/tabelas/03_tabela_mestre/rotulos_sem_imagem.csv`.

No split original, a MobileNetV2 com recortes foi o modelo visual mais
equilibrado no threshold 0,50, com balanced accuracy 0,6402 e MCC 0,2738. Na
validação por tratamento, esse desempenho não se sustentou: balanced accuracy
micro 0,4460 e MCC -0,1061. O Random Forest com threshold escolhido somente na
validação interna obteve a melhor combinação amostral entre recall e
especificidade dentre os candidatos visuais considerados para referência:
recall 0,8531, especificidade 0,1934, F1 0,7205, balanced accuracy 0,5233 e MCC
0,0612. Evidências:
`saidas/tabelas/07_classificacao_final/comparacao_final_classificacao.csv` e
`saidas/tabelas/07_classificacao_final/validacao_tratamento/resumo_generalizacao_por_tratamento.csv`.

O baseline de metadados superou os modelos visuais em algumas métricas micro,
mas não usa pixels e sua balanced accuracy macro foi 0,5000. Ele deve ser
interpretado como diagnóstico de viés de origem, lote ou tratamento, não como
modelo candidato ao aplicativo. Evidências:
`saidas/tabelas/06_modelos/metadados/conclusao_vies_metadados.txt` e
`docs/relatorio_classificacao_cientifica.md`.

A triagem oficial classificou as 703 sementes como alto risco, sem nenhuma
semente em baixo risco ou incerteza. A regra atingiu recall 1,0, mas com
precisão 0,6102 e utilidade de baixo risco igual a zero. Portanto, não foi
encontrada uma zona segura de liberação. Evidências:
`saidas/tabelas/08_triagem/comparacao_scores_triagem.csv`,
`saidas/tabelas/08_triagem/score_triagem_recomendado.csv` e
`docs/relatorio_triagem_preventiva.md`.

### Conclusão científica central refinada

> O estudo de viabilidade mostrou que, no conjunto e no protocolo avaliados,
> imagens RGB registradas no início dos experimentos não sustentaram
> classificação confiável e generalizável da contaminação observada
> posteriormente em tratamentos não vistos. Houve sinal preditivo limitado no
> split original, mas ele foi instável entre tratamentos e fortemente
> confundido por diferenças de domínio, origem e tratamento. O resultado não
> prova que inexista qualquer sinal visual inicial; demonstra que os dados, a
> padronização de aquisição e os modelos atualmente disponíveis são
> insuficientes para uma decisão operacional segura.

Essa redação é mais precisa que afirmar simplesmente que as imagens “não
funcionam”: preserva o resultado negativo do estudo de viabilidade sem
extrapolar além dos dados.

## 3. Estrutura real do repositório

| Caminho | Conteúdo real | Situação |
|---|---|---|
| `dados_originais/imagens/` | imagens dos experimentos Micro-ondas, Piloto e TESTE 2 | presente, ignorado pelo Git |
| `dados_originais/planilhas/` | três planilhas-fonte | presente, ignorado pelo Git |
| `scripts/preparacao/` | inventário, rótulos, tabela mestre, dataset e conferência | versionado |
| `scripts/baseline/` | ResNet18 com imagem inteira | versionado |
| `scripts/caixas_yolo/` | caixas, ajustes, dataset YOLO, treino e avaliação | versionado |
| `scripts/recortes/` | recortes, atributos, modelos, comparação e validação externa | versionado |
| `scripts/triagem/` | integração, thresholds, scores, comparação e relatório | versionado |
| `docs/` | documentação científica e figuras finais | versionado |
| `saidas/` | tabelas, figuras, datasets derivados, checkpoints e logs | presente, ignorado pelo Git |
| `entrega_final/` | DOCX e PPTX preliminares | não rastreado e não oficial |
| `outputs/` | diretório vazio | sem função documentada |
| `active`, `conda`, `frozen` | arquivos vazios de zero byte | aparentemente acidentais, não utilizados |
| `yolo11n.pt` | peso-base usado pelo YOLO | presente e ignorado |
| `yolo26n.pt` | peso sem uso encontrado na pipeline | presente e ignorado |

O `.gitignore` exclui `dados_originais/`, `saidas/`, checkpoints e documentos
finais. Isso mantém o Git leve, mas significa que o commit e o GitHub, sozinhos,
não preservam os dados, resultados, modelos ou entregáveis necessários para
reproduzir a pesquisa. Evidências: `.gitignore` e `README.md`.

### Planilhas originais

| Arquivo | Estrutura auditada | Papel |
|---|---|---|
| `dados_originais/planilhas/23-04-2026 - Novos Índices.xlsx` | 1 aba, 223 linhas, 23 colunas e 666 células com fórmulas | rótulos e índices do experimento Micro-ondas |
| `dados_originais/planilhas/Piloto -Contaminacao-Germinacao-Umidade.xlsx` | abas `Umidade Amostra` e `Contaminação`, 83 e 82 linhas não vazias | dados do Piloto |
| `dados_originais/planilhas/TABELA PARA ANALISE - TESTE 2.xlsx` | 1 aba e 4.801 linhas não vazias | germinação e contaminação do TESTE 2 |

As planilhas são fontes primárias de dados. Fórmulas existentes não foram
recalculadas nesta auditoria.

### Inventário físico

Foram localizados 3.565 arquivos JPG, 5 JPEG, 102 CSV, 55 scripts Python, 35
PNG, 19 checkpoints `pt`, 10 JSON e 3 XLSX. O inventário científico reconhece
748 imagens experimentais, incluindo quatro imagens metodológicas em
`TESTE 2/metd`. Evidência principal:
`saidas/tabelas/01_inventario/resumo_inventario.csv`.

## 4. Cronologia reconstruída pelos commits

Os hashes abaixo são abreviados para leitura; o commit atual completo está
registrado na seção 1.

| Data | Commit(s) | Evolução confirmada |
|---|---|---|
| 01/06/2026 | `e3a2765` | criação da estrutura, preparação dos dados e baseline inicial |
| 01/06/2026 | `cf09c93`, `14473d7` | introdução do YOLO, caixas automáticas e ajustes manuais |
| 02/06/2026 | `da4aea1`, `d1aaea8`, `4f5e2f5` | refatoração do YOLO, análise de erros, recortes e ResNet18 com recortes |
| 02/06/2026 | `90a77d5` a `88bf6dc` | primeira tentativa de triagem conservadora e calibração |
| 04/06/2026 | `b5669df`, `3b9bdb6` | criação do baseline de metadados |
| 04/06/2026 | `6128e14`, `2e8de37`, `6a1d2ab` | consolidação da pipeline e reorganização das saídas |
| 04–05/06/2026 | `afc6393`, `741a313`, `dd5396a`, `0f64c38` | extração de atributos visuais e modelos clássicos |
| 05/06/2026 | `12fa6c8`, `e94a072`, `e82ac6e` | MobileNetV2, garantia de pesos ImageNet e comparação final |
| 05–07/06/2026 | `770ead7`, `b9684a0`, `a9cbb9d` | validação por tratamento, modularização e correção de import |
| 08/06/2026 | `c8d6690` a `8890cd3` | geração e revisão do relatório científico da classificação |
| 08/06/2026 | `9418a71`, `28316ef` | adequação ao escopo e fechamento científico |
| 08/06/2026 | `d11cc9c`, `91d0dec` | inclusão e validação externa de k-NN e LDA |
| 08/06/2026 | `00a314c`, `a37f680` | redesenho da triagem e calibração interna por grupo |
| 08/06/2026 | `a0e7e40` a `c3b951a` | revisão final do relatório e das figuras de triagem |

O histórico confirma uma mudança metodológica importante: a avaliação deixou
de depender apenas do split original e passou a testar generalização entre
tratamentos. Essa mudança, consolidada a partir de `770ead7`, é a principal
razão para a conclusão negativa de viabilidade operacional.

## 5. Inventário das etapas executadas

| Etapa | Situação | Script principal | Saída/evidência |
|---|---|---|---|
| 1. Inventário das imagens | concluída | `scripts/preparacao/00_inventario_imagens.py` | `saidas/tabelas/01_inventario/inventario_imagens.csv` |
| 2. Leitura das planilhas | concluída | `scripts/preparacao/01_ler_planilhas_rotulos.py` | `saidas/tabelas/02_planilhas_rotulos/` |
| 3. Criação dos rótulos | concluída | `scripts/preparacao/02_criar_rotulos_planilhas.py` | rótulos consolidados em `02_planilhas_rotulos/` |
| 4. Tabela mestre | concluída com faltantes explícitos | `scripts/preparacao/03_criar_tabela_mestre.py` | `saidas/tabelas/03_tabela_mestre/tabela_mestre.csv` |
| 5. Dataset binário | concluída | `scripts/preparacao/04_criar_dataset_binario.py` | `saidas/dataset_binario/` |
| 6. Split treino/validação/teste | concluída e reutilizada | mesmo script da etapa 5 | `saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv` |
| 7. ResNet18 com imagem inteira | concluída | `scripts/baseline/06_treinar_baseline.py`, `07_avaliar_modelo.py` | `saidas/tabelas/06_modelos/baseline/` |
| 8. Caixas automáticas | concluída | scripts `08` a `13` em `scripts/caixas_yolo/` | `saidas/tabelas/05_caixas_yolo/caixas_automaticas.csv` |
| 9. YOLO | concluída | scripts `14` a `17` | `saidas/yolo_runs/sementes_yolo_caixas_auto/` |
| 10. Recortes | concluída | pipeline de avaliação YOLO e recortes | `saidas/dataset_recortado/` |
| 11. ResNet18 com recortes | concluída | scripts `18` a `21` | `saidas/tabelas/06_modelos/recortes/` |
| 12. Atributos visuais | concluída | `scripts/recortes/22_extrair_atributos_visuais_recortes.py` | `saidas/tabelas/06_modelos/classicos/features_classicos.csv` |
| 13. Random Forest | concluída | `scripts/recortes/23_treinar_avaliar_classicos_recortes.py` | tabelas de `06_modelos/classicos/` |
| 14. SVM RBF | concluída | mesmo script | tabelas de `06_modelos/classicos/` |
| 15. k-NN | concluída posteriormente | mesmo script | comparação final e validação externa |
| 16. LDA | concluída posteriormente | mesmo script | comparação final e validação externa |
| 17. MobileNetV2 | concluída | scripts `24` e `25` | `saidas/tabelas/06_modelos/mobilenetv2/` |
| 18. Baseline de metadados | concluída, diagnóstico apenas | `scripts/recortes/26_baseline_metadados_classificacao.py` | `saidas/tabelas/06_modelos/metadados/` |
| 19. Comparação final | concluída | `scripts/recortes/27_comparar_classificacao_final.py` | `comparacao_final_classificacao.csv` |
| 20. Validação por experimento/tratamento | concluída para sete modelos/controles | script `28` e pacote `validacao_tratamento/` | `saidas/tabelas/07_classificacao_final/validacao_tratamento/` |
| 21. Análise do T6 | concluída por filtros e inspeção, sem artefato dedicado | pipeline geral | imagens, split e métricas por grupo T6 |
| 22. Análise dos erros | concluída | scripts `17` e `21` | `saidas/conferencia_yolo/` e `saidas/conferencia_recortes/erros/` |
| 23. Triagem preventiva | concluída | scripts `30` a `32` | `saidas/tabelas/08_triagem/` |
| 24. Calibração e comparação da triagem | concluída | scripts `33` a `35` | manifestos, scores e relatório final |
| 25. Aplicativo | intencionalmente não implementado | decisão científica | `docs/adequacao_escopo_proposta.md` |

## 6. Fontes oficiais recomendadas

### Fontes científicas principais

1. `docs/relatorio_classificacao_cientifica.md`: síntese científica final da
   classificação.
2. `docs/relatorio_triagem_preventiva.md`: síntese final da triagem.
3. `docs/adequacao_escopo_proposta.md`: justificativa de adequação ao escopo e
   não desenvolvimento do aplicativo.
4. `saidas/tabelas/07_classificacao_final/comparacao_final_classificacao.csv`:
   resultados consolidados do split original.
5. `saidas/tabelas/07_classificacao_final/validacao_tratamento/resumo_generalizacao_por_tratamento.csv`:
   resultados micro e macro da validação por tratamento.
6. `saidas/tabelas/07_classificacao_final/validacao_tratamento/metricas_validacao_por_tratamento.csv`:
   matrizes e métricas por fold.
7. `saidas/tabelas/08_triagem/comparacao_scores_triagem.csv` e
   `score_triagem_recomendado.csv`: decisão operacional final.

### Fontes de rastreabilidade

- `saidas/tabelas/07_classificacao_final/relatorio/manifesto_experimento_final.json`
- `saidas/tabelas/08_triagem/manifesto_thresholds_triagem.json`
- `saidas/tabelas/08_triagem/manifesto_scores_triagem.json`
- `saidas/tabelas/08_triagem/manifesto_comparacao_triagem.json`
- `saidas/modelos/config_baseline_resnet18.json`
- `saidas/modelos/config_recortes_resnet18.json`
- `saidas/modelos/config_mobilenetv2_recortes.json`
- `saidas/yolo_runs/sementes_yolo_caixas_auto/args.yaml`

### Fontes primárias dos dados

- as três planilhas em `dados_originais/planilhas/`;
- as imagens em `dados_originais/imagens/`;
- a tabela mestre e o split consolidado.

### Arquivos que não devem ser tratados como fontes finais

- `docs/relatorio_contexto_chatgpt.md`: documento histórico anterior à
  validação externa final;
- `entrega_final/Relatorio_Final_Projeto_Sementes.docx`: não rastreado, com
  placeholders e sem bibliografia formal;
- `entrega_final/Apresentacao_Final_Projeto_Sementes.pptx`: não rastreada e
  com identificação incompleta;
- tabelas de sensibilidade com features não oficiais: úteis apenas como análise
  exploratória;
- checkpoints isolados sem o manifesto/configuração correspondente.

## 7. Resultados finais consolidados

### Dados e split

| Conjunto | Contaminada | Não contaminada | Total |
|---|---:|---:|---:|
| Treino | 299 | 192 | 491 |
| Validação | 65 | 41 | 106 |
| Teste | 65 | 41 | 106 |
| Total | 429 | 274 | 703 |

Fonte: `saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv`.

### Integridade de dados

- imagens problemáticas ou ilegíveis: 0;
- duplicatas na chave composta de inventário: 0;
- duplicatas de rótulo: 0;
- imagens sem rótulo: 45;
- rótulos sem imagem: 79;
- caixas/linhas do dataset YOLO: 703;
- ajustes manuais de caixa: 80, todos no T6.

Fontes:
`saidas/tabelas/01_inventario/imagens_problematicas.csv`,
`saidas/tabelas/03_tabela_mestre/duplicatas_inventario_chave.csv`,
`saidas/tabelas/02_planilhas_rotulos/duplicatas_rotulos.csv`,
`saidas/tabelas/05_caixas_yolo/relatorio_dataset_yolo.csv` e
`saidas/tabelas/05_caixas_yolo/caixas_ajustes_manuais.csv`.

### Síntese dos resultados

- o split original permitiu desempenho visual moderado, principalmente para
  MobileNetV2 no threshold 0,50;
- a redução de thresholds elevou recall, mas quase eliminou a especificidade;
- SVM no split original reproduziu o controle “sempre contaminada”;
- o baseline de metadados indicou forte associação entre classe e contexto
  experimental;
- a validação por tratamento reduziu ou tornou instável o desempenho;
- nenhum modelo sustentou simultaneamente recall alto, especificidade útil e
  estabilidade entre tratamentos;
- a triagem não encontrou baixo risco seguro.

## 8. Divergências e inconsistências encontradas

| Divergência | Evidência | Interpretação recomendada |
|---|---|---|
| YOLO: `BATCH = 6` no script, mas `batch: 4` na execução | `scripts/caixas_yolo/15_treinar_yolo.py`; `saidas/yolo_runs/sementes_yolo_caixas_auto/args.yaml` | usar `batch: 4` para descrever a execução auditada; registrar que o script mudou depois ou divergiu |
| Caminho antigo do split no ResNet18 de recortes | `saidas/modelos/config_recortes_resnet18.json` aponta para `saidas/tabelas/divisao...`, inexistente | o split efetivo é `saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv` |
| Configuração final da validação lista apenas k-NN e LDA | `config_validacao_por_tratamento.json` | arquivo foi sobrescrito pela última execução retomável; o manifesto e o CSV final confirmam sete modelos/controles |
| Não existe tag `relatorio-final-v1` | Git | criar somente após aprovação e congelamento dos artefatos |
| Ambiente sem versões fixadas | `environment.yml` | reprodução aproximada, não exata |
| `torch` e `torchvision` não constam no ambiente | imports dos scripts e `environment.yml` | instalação é delegada ao link externo do PyTorch |
| Saídas, dados e checkpoints estão fora do Git | `.gitignore` | GitHub não é um pacote reprodutível completo |
| T6 não possui tabela/relatório dedicado | métricas gerais permitem filtro, mas não há artefato específico | gerar posteriormente uma tabela derivada, sem recalibrar modelos |
| DOCX/PPTX preliminares não estão versionados e têm placeholders | `entrega_final/` | não usar como fonte de números ou identificação |
| `crossfit` permanece em nomes de arquivos | `docs/relatorio_triagem_preventiva.md` | significa leave-one-group-out com calibração interna, não cross-fitting estatístico clássico |
| Caixas do YOLO são pseudo-rótulos automáticos | scripts `08`–`14` e `caixas_automaticas.csv` | não descrevê-las como ground truth manual de detecção |
| Arquivos vazios e peso sem uso | `active`, `conda`, `frozen`, `yolo26n.pt` | registrar como resíduos não utilizados; não apagar nesta etapa |
| Ausência de proposta original e bibliografia | busca no repositório | `adequacao_escopo_proposta.md` é análise de escopo, não substitui a proposta oficial |

Nenhuma dessas divergências altera a conclusão científica central, mas elas
afetam rastreabilidade, redação metodológica e reprodução exata.

## 9. Comparação dos modelos

### Split original, cenário fixo ou regra principal de cada modelo

| Modelo | Threshold/regra | Recall | Especificidade | F1 | Balanced accuracy | MCC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Controle sempre contaminada | sempre positivo | 1,0000 | 0,0000 | 0,7602 | 0,5000 | 0,0000 | 41 | 0 |
| ResNet18 imagem inteira | 0,50 | 0,9385 | 0,0488 | 0,7394 | 0,4936 | -0,0269 | 39 | 4 |
| YOLO classificação | melhor detecção, conf. 0,25 | 0,8615 | 0,1951 | 0,7273 | 0,5283 | 0,0752 | 33 | 9 |
| ResNet18 recortes | 0,50 | 0,8308 | 0,3171 | 0,7347 | 0,5739 | 0,1720 | 28 | 11 |
| Random Forest | 0,50 | 0,8615 | 0,3171 | 0,7517 | 0,5893 | 0,2145 | 28 | 9 |
| SVM RBF | 0,50 | 1,0000 | 0,0000 | 0,7602 | 0,5000 | 0,0000 | 41 | 0 |
| k-NN | 0,50 | 0,9385 | 0,1707 | 0,7625 | 0,5546 | 0,1744 | 34 | 4 |
| LDA | 0,50 | 0,7077 | 0,3659 | 0,6715 | 0,5368 | 0,0767 | 26 | 19 |
| MobileNetV2 recortes | 0,50 | 0,6462 | 0,6341 | 0,6885 | 0,6402 | 0,2738 | 15 | 23 |
| Metadados | 0,50 | 0,9385 | 0,3902 | 0,8079 | 0,6644 | 0,4092 | 25 | 4 |

Fonte:
`saidas/tabelas/07_classificacao_final/comparacao_final_classificacao.csv`.
O baseline de metadados é diagnóstico de viés e não candidato visual.

O F1 de SVM e do controle sempre contaminada mostra por que F1 e recall não
podem ser usados isoladamente. Ambos atingem recall 1,0 sem reconhecer nenhuma
semente não contaminada.

### Efeito da calibração de threshold

- Random Forest: threshold 0,40 elevou recall para 0,9692, mas reduziu
  especificidade para 0,0732.
- MobileNetV2: threshold 0,38 elevou recall para 0,8769, mas reduziu
  especificidade para 0,1707.
- k-NN: threshold de prioridade de recall 0,26 produziu especificidade zero.
- LDA: threshold 0,01 produziu recall 0,9846 e especificidade 0,0244.

Esses resultados estão no mesmo CSV consolidado e demonstram o custo
operacional de reduzir falsos negativos: crescimento acentuado de falsos
positivos.

## 10. Modelo recomendado como referência principal

### Recomendação

Adotar o **Random Forest com 127 atributos do conjunto
`principal_normalizado` e threshold selecionado exclusivamente na validação
interna de cada fold** como principal referência científica para a discussão
de generalização.

### Justificativa

1. Na validação por tratamento, sua configuração calibrada obteve recall
   0,8531, especificidade 0,1934, F1 0,7205, balanced accuracy 0,5233 e MCC
   0,0612.
2. Produziu 63 falsos negativos e 221 falsos positivos em 703 amostras, uma
   relação menos degenerada que os modelos que tenderam a prever tudo como
   contaminado.
3. Usa atributos normalizados e exclui dimensões absolutas e textura dependente
   da resolução original, reduzindo parte do risco de explorar diferenças
   triviais de aquisição.
4. É computacionalmente mais simples e auditável que as redes profundas.
5. Ainda assim, seu desempenho é apenas ligeiramente superior ao acaso em
   balanced accuracy e MCC, portanto ele é uma **referência científica**, não
   um modelo aprovado para uso.

Evidências:
`saidas/tabelas/06_modelos/classicos/features_classicos.csv`,
`saidas/tabelas/07_classificacao_final/validacao_tratamento/resumo_generalizacao_por_tratamento.csv`
e `docs/relatorio_classificacao_cientifica.md`.

### Alternativas sob outros critérios

- **MobileNetV2 no threshold 0,50:** melhor referência se o foco for equilíbrio
  no split original, maior especificidade e arquitetura compatível com
  dispositivos móveis. Não generalizou no protocolo por tratamento.
- **SVM no threshold 0,50:** apresentou médias macro relativamente melhores
  entre grupos, mas desempenho micro quase neutro e comportamento próximo do
  controle positivo em outros cenários.
- **k-NN calibrado:** maior recall e MCC micro externo que o Random Forest, mas
  especificidade de apenas 0,0547, insuficiente para separar baixo risco.

A escolha do Random Forest é defensável como eixo narrativo, mas não representa
um “vencedor” operacional.

## 11. Avaliação da classificação direta

O split original é útil para comparar implementações sob a mesma divisão, mas
mistura sementes de contextos experimentais semelhantes em desenvolvimento e
teste. Por isso, ele pode superestimar a capacidade de generalizar para um novo
tratamento.

A MobileNetV2 foi o modelo visual mais equilibrado no teste original. Sua
matriz foi TN=26, FP=15, FN=23 e TP=42. O custo do equilíbrio foi recall menor,
0,6462. A redução do threshold recuperou recall, mas tornou a especificidade
inadequada. Evidências:
`saidas/tabelas/06_modelos/mobilenetv2/metricas_mobilenetv2_recortes_teste.csv`
e `saidas/figuras/curva_threshold_mobilenetv2_recortes_validacao.png`.

O ResNet18 de imagem inteira e o YOLO apresentaram baixa capacidade de
reconhecer a classe não contaminada. O ResNet18 com recortes e os atributos
clássicos melhoraram o equilíbrio, confirmando que isolar a semente reduziu
parte do ruído de fundo, mas não resolveu o problema de generalização.

As caixas foram produzidas por processamento de imagem e usadas como
pseudo-rótulos para o YOLO. O experimento avalia uma pipeline de localização e
classificação, não um detector treinado contra anotações humanas independentes.

## 12. Validação por tratamento e generalização

O protocolo deixou cada um dos 12 grupos `experimento_tratamento` de fora uma
vez como teste. Um segundo grupo inteiro foi usado internamente para seleção de
threshold. O split original permaneceu apenas para auditoria. Evidências:
`scripts/recortes/validacao_tratamento/folds.py`,
`folds_validacao_por_tratamento.csv` e
`docs/relatorio_classificacao_cientifica.md`.

Essa validação é mais rigorosa que o split original, mas ainda não é validação
externa em uma coleta independente: todos os grupos pertencem ao mesmo acervo
do projeto.

### Resultados externos representativos

| Modelo/cenário | Recall | Especificidade | F1 | Balanced accuracy | MCC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
| Random Forest, threshold interno de melhor F1 | 0,8531 | 0,1934 | 0,7205 | 0,5233 | 0,0612 | 53/221/63/366 |
| k-NN, threshold interno de melhor F1 | 0,9837 | 0,0547 | 0,7604 | 0,5192 | 0,1076 | 15/259/7/422 |
| SVM, threshold 0,50 | 0,8182 | 0,1934 | 0,7013 | 0,5058 | 0,0145 | 53/221/78/351 |
| MobileNetV2, threshold 0,50 | 0,5198 | 0,3723 | 0,5413 | 0,4460 | -0,1061 | 102/172/206/223 |
| LDA, threshold 0,50 | 0,4709 | 0,3978 | 0,5075 | 0,4343 | -0,1282 | 109/165/227/202 |
| Metadados, threshold 0,50 | 0,9510 | 0,1788 | 0,7684 | 0,5649 | 0,2115 | 49/225/21/408 |

Fonte:
`saidas/tabelas/07_classificacao_final/validacao_tratamento/resumo_generalizacao_por_tratamento.csv`.

O baseline de metadados teve balanced accuracy macro 0,5000 e MCC macro 0,0,
apesar do resultado micro. Essa diferença indica que o resultado depende do
tamanho e da composição dos grupos, reforçando seu papel de diagnóstico de
viés.

### Estabilidade

Os grupos variam de 3 amostras no T4 a 115 em
`micro_ondas__controle`. Métricas macro são sensíveis aos grupos pequenos, e
métricas micro são dominadas pelos grupos maiores. Nenhuma das duas deve ser
interpretada isoladamente.

O Random Forest calibrado teve balanced accuracy macro média 0,5103 ± 0,0674 e
MCC macro 0,0220 ± 0,1278. O SVM fixo teve balanced accuracy macro 0,5582 ±
0,0722 e MCC 0,1098 ± 0,1305, mas seu desempenho micro permaneceu quase neutro.
Essa divergência entre agregações deve aparecer no relatório final.

### Custo computacional registrado

Somando uma execução por fold:

- MobileNetV2: 18.939,61 s, aproximadamente 5 h 15 min 40 s;
- Random Forest: 2.795,88 s, aproximadamente 46 min 36 s;
- demais modelos clássicos e metadados: menos de 1 min no total;
- total de treino registrado: aproximadamente 6 h 03 min, sem contar
  preparação, I/O e geração de figuras.

Fonte:
`metricas_validacao_por_tratamento.csv`, coluna
`tempo_treino_segundos`.

## 13. Análise do T6

O T6 contém 80 sementes, 49 contaminadas e 31 não contaminadas. No split
original foram 55 amostras de treino, 11 de validação e 14 de teste. Evidências:
`resumo_treinavel.csv` e `divisao_treino_validacao_teste.csv`.

As imagens do T6 diferem visualmente dos demais grupos:

- resolução típica observada de 627 × 836 pixels;
- presença de régua, pinça e enquadramento mais aberto;
- semente ocupando menor fração da imagem;
- recortes finais muito pequenos em alguns casos.

Como contraste, `dados_originais/imagens/TESTE 2/T5/f4.jpg` possui
aproximadamente 12.000 × 9.000 pixels, enquanto
`dados_originais/imagens/TESTE 2/T6/f4.jpg` possui 627 × 836. O recorte
`saidas/dataset_recortado/contaminada/TESTE_2__T6__f4.jpg` tem apenas 6.364
bytes. Foram aplicados 80 ajustes manuais de caixa, todos no T6.

### T6 como grupo externo, threshold 0,50

| Modelo | Recall | Especificidade | F1 | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,9796 | 0,0323 | 0,7559 | 0,5059 | 0,0370 |
| SVM | 1,0000 | 0,0000 | 0,7597 | 0,5000 | 0,0000 |
| k-NN | 0,9592 | 0,0323 | 0,7460 | 0,4957 | -0,0219 |
| MobileNetV2 | 0,6122 | 0,2903 | 0,5941 | 0,4513 | -0,0995 |
| LDA | 0,2245 | 0,6774 | 0,3143 | 0,4510 | -0,1086 |
| Metadados | 1,0000 | 0,0000 | 0,7597 | 0,5000 | 0,0000 |

Fonte:
`saidas/tabelas/07_classificacao_final/validacao_tratamento/metricas_validacao_por_tratamento.csv`,
filtro `grupo_externo = teste_2__t6`.

O T6 evidencia mudança de domínio, mas os dados não permitem atribuir
causalidade exclusiva à resolução, à régua, à pinça ou ao enquadramento. Esses
fatores ocorrem juntos. O relatório deve dizer que são explicações plausíveis,
não causas comprovadas.

Não existe tabela científica dedicada ao T6. Os números podem ser reproduzidos
por filtro dos CSV gerais, mas a ausência de um artefato específico é uma
lacuna de apresentação e rastreabilidade.

## 14. Avaliação da triagem

A triagem integrou Random Forest, SVM, k-NN, LDA e MobileNetV2. A estratégia
oficial `consenso_pre_especificado` foi definida antes da comparação externa:

- baixo risco: todos os modelos abaixo dos thresholds baixos seguros;
- alto risco: ao menos um modelo acima do threshold alto;
- incerto: demais casos;
- sem threshold baixo seguro, o baixo risco é suspenso.

O threshold baixo exigia zero falso negativo na validação interna e utilidade
mínima para não contaminadas. O threshold alto foi escolhido por F1, com
desempates de recall, precisão e menor FP. Evidências:
`scripts/triagem/33_calibrar_thresholds_triagem.py`,
`manifesto_thresholds_triagem.json` e
`docs/relatorio_triagem_preventiva.md`.

### Resultado oficial

| Total | Baixo risco | Incerto | Alto risco | Recall alto risco | Precisão alto risco |
|---:|---:|---:|---:|---:|---:|
| 703 | 0 | 0 | 703 | 1,0000 | 0,6102 |

Nenhum modelo individual criou baixo risco. As estratégias individuais apenas
redistribuíram parte das amostras entre alto risco e incerto. A triagem oficial
é equivalente, em utilidade, a encaminhar todas as sementes para revisão; não
reduz carga operacional e não autoriza liberação automática.

O termo `crossfit` nos nomes dos arquivos não deve ser apresentado como
cross-fitting estatístico. O protocolo real é leave-one-treatment-out com
calibração interna por grupo.

## 15. Problemas encontrados e soluções adotadas

| Problema | Tratamento aplicado | O que revelou cientificamente |
|---|---|---|
| nomes e IDs repetidos | chave composta com experimento, pasta e ID | o ID isolado não identifica uma semente em todo o projeto |
| imagens sem rótulo | lista explícita e exclusão do conjunto treinável | há imagens extras/metodológicas que não podem receber classe por inferência |
| rótulos sem imagem | lista explícita e status `sem_imagem` | a cobertura experimental é incompleta, sobretudo no T4 |
| T4 com 80 rótulos planejados e apenas 3 imagens | apenas as 3 associações válidas foram usadas | o grupo externo T4 é estatisticamente frágil |
| 15 imagens em `Piloto/Contaminadas C` sem rótulo | mantidas fora do treino | duplicatas ou extras não foram incorporados silenciosamente |
| diferenças de fundo, iluminação e enquadramento | recortes e atributos normalizados | reduzir fundo ajudou, mas não eliminou viés de domínio |
| T6 com padrão visual próprio | 80 ajustes manuais e análise por grupo | aquisição não padronizada prejudica transferência |
| baixa especificidade | comparação com balanced accuracy e MCC | recall alto pode ser produzido por previsão excessiva da classe positiva |
| queda de recall ao elevar threshold | curvas de threshold | existe conflito real entre segurança contra FN e utilidade para não contaminadas |
| muitos falsos positivos | grades de erros e matrizes | o modelo não encontra separação visual operacionalmente útil |
| metadados competitivos | baseline sem pixels | parte do sinal está associada a tratamento/origem, não necessariamente à semente |
| split original melhor que validação por tratamento | protocolo leave-one-group-out | o split aleatório compartilha contexto e superestima generalização |
| ausência de baixo risco | regra conservadora de triagem | não existe região de confiança suficiente para liberar sementes |
| alto custo da validação externa profunda | execução retomável e modularizada | o custo cresceu sem produzir ganho de robustez correspondente |
| imagens RGB iniciais | comparação multimodelo e por tratamento | sinais pré-sintomáticos, se presentes, são fracos ou confundidos neste protocolo |

Fontes centrais:
`tabela_mestre.csv`, `imagens_sem_rotulo.csv`,
`rotulos_sem_imagem.csv`, `comparacao_final_classificacao.csv`,
`metricas_validacao_por_tratamento.csv` e os dois relatórios científicos em
`docs/`.

## 16. Justificativa para o aplicativo não ter sido desenvolvido

O aplicativo proposto exigiria um classificador capaz de separar sementes de
alto e baixo risco com comportamento previsível em novos tratamentos. Os
resultados não atendem a esse requisito:

1. a melhor configuração do split original perdeu desempenho na validação por
   tratamento;
2. modelos de alto recall tiveram especificidade muito baixa;
3. o baseline de metadados revelou risco de viés experimental;
4. o T6 demonstrou sensibilidade a mudanças de aquisição;
5. a triagem não encontrou nenhuma amostra de baixo risco seguro.

Implementar um aplicativo com esses resultados poderia transformar um protótipo
exploratório em uma interface com aparência de decisão agronômica confiável,
sem suporte científico para isso. A decisão de não implementar é coerente com
um estudo de viabilidade e está documentada em
`docs/adequacao_escopo_proposta.md`.

Um eventual demonstrador futuro só deveria ser apresentado como ferramenta de
pesquisa, sem liberação automática e após nova coleta padronizada e validação
independente.

## 17. Figuras selecionadas

Todos os caminhos são relativos à raiz `C:\Projetos\sementes_ia`.

| Nº | Título sugerido e seção | Caminho exato | Demonstração e necessidade | Combinação | Adequação |
|---:|---|---|---|---|---|
| 1 | Exemplos das duas classes; Dados | `saidas/amostras_conferencia/amostras_contaminada.png` e `saidas/amostras_conferencia/amostras_nao_contaminada.png` | sobreposição visual entre classes e diversidade de origens | painel A/B | 1404×2360 cada; usar página inteira |
| 2 | Contraste entre aquisição T5 e T6; Dados/T6 | `dados_originais/imagens/TESTE 2/T5/f4.jpg` e `dados_originais/imagens/TESTE 2/T6/f4.jpg` | diferença de resolução, escala, régua, pinça e enquadramento | painel A/B | adequada; reduzir T5 sem perder nitidez |
| 3 | Caixa ajustada e recorte no T6; Métodos/T6 | `saidas/conferencia_caixas/imagens/contaminada/TESTE_2__T6__f4.jpg` e `saidas/dataset_recortado/contaminada/TESTE_2__T6__f4.jpg` | efeito do ajuste e baixa informação espacial do recorte | painel A/B | anotação legível; baixa resolução do recorte é parte do achado |
| 4 | Matriz de confusão da MobileNetV2; Classificação direta | `saidas/figuras/matriz_confusao_mobilenetv2_recortes_teste.png` | melhor equilíbrio visual no split original | não | 900×750, adequada |
| 5 | Matriz de confusão da ResNet18 com recortes; Classificação direta | `saidas/figuras/matriz_confusao_recortes_resnet18_teste.png` | efeito dos recortes e persistência de FP | não | 900×750, adequada |
| 6 | Trade-off de threshold da MobileNetV2; Classificação direta | `saidas/figuras/curva_threshold_mobilenetv2_recortes_validacao.png` | perda de especificidade ao priorizar recall | não | 1200×750, adequada |
| 7 | Comparação dos modelos no split original; Resultados | `docs/figuras/classificacao/metricas_split_original.png` | comparação conjunta de recall, especificidade, F1, BA e MCC | não | 1600×960; rótulos longos exigem largura total |
| 8 | Desempenho micro na validação por tratamento; Generalização | `docs/figuras/classificacao/metricas_validacao_externa_micro.png` | desempenho fora do grupo de desenvolvimento | não | 1600×960, adequada |
| 9 | Queda do split original para validação por tratamento; Generalização | `docs/figuras/classificacao/comparacao_split_original_vs_validacao_externa.png` | principal evidência de perda de generalização | não | 1600×960, adequada |
| 10 | Variação entre tratamentos; Generalização | `docs/figuras/classificacao/variacao_entre_tratamentos.png` | instabilidade dos modelos entre grupos | não | 1600×960; legenda deve alertar para T4 com n=3 |
| 11 | Falsos positivos da ResNet18 com recortes; Discussão de erros | `saidas/conferencia_recortes/erros/falsos_positivos_recortes_threshold_0_50.png` | diversidade das sementes não contaminadas marcadas como risco | não | 1754×2302; usar página inteira |
| 12 | Falsos negativos da ResNet18 com recortes; Discussão de erros | `saidas/conferencia_recortes/erros/falsos_negativos_recortes_threshold_0_50.png` | casos contaminados não reconhecidos | não | 1754×1195, adequada |
| 13 | Distribuição final das decisões de triagem; Triagem | `docs/figuras/triagem/distribuicao_decisoes_triagem.png` | inexistência de baixo risco no consenso | não | 1760×960; substituir `estrategia_legivel` na versão editorial |
| 14 | Triagem por grupo experimental; Triagem | `docs/figuras/triagem/triagem_por_grupo_consenso.png` | todas as amostras de todos os grupos em alto risco | não | 1920×960; rótulos longos exigem largura total |

Não existe figura pronta de fluxograma completo nem gráfico simples de
distribuição das classes. Eles são desejáveis, mas não devem ser improvisados
nesta etapa. A geração deve ocorrer somente após aprovação, usando os números
congelados da tabela mestre e do split.

## 18. Situação da reprodutibilidade

| Item | Classificação | Evidência | Lacuna |
|---|---|---|---|
| Estrutura de pastas | completo | `README.md` e diretórios | nenhuma relevante |
| Dados originais locais | parcialmente documentado | `dados_originais/` e planilhas | não versionados, sem checksum ou link público |
| Ambiente Conda | parcialmente documentado | `environment.yml` | versões não fixadas |
| Python | completo | `python=3.11` | versão patch ausente |
| Bibliotecas principais | parcialmente documentado | `environment.yml` | `torch` e `torchvision` ausentes; demais sem versão |
| Hardware | parcialmente documentado | `docs/relatorio_contexto_chatgpt.md` | informação histórica, sem log por execução |
| GPU/CUDA | parcialmente documentado | README e configs CUDA | versões de CUDA, cuDNN e driver ausentes |
| Configuração dos modelos | parcialmente documentado | JSON, YAML e históricos | divergências registradas na seção 8 |
| Sementes aleatórias | parcialmente documentado | seed 42 na maioria; seed 0 no YOLO | nem todos os scripts registram todas as fontes de aleatoriedade |
| Ordem dos scripts | completo | `README.md`, numeração 00–35 | nenhuma relevante |
| Comandos principais | completo | `README.md` | comandos dependem de ambiente não congelado |
| Entradas e saídas | completo localmente | README, scripts e manifestos | saídas estão fora do Git |
| Split original | completo | `divisao_treino_validacao_teste.csv` | arquivo ignorado pelo Git |
| Validação por tratamento | parcialmente documentado | CSVs, pacote e manifesto | config final foi sobrescrita pelo último subconjunto |
| Checkpoints | completo localmente | `saidas/modelos/` e `saidas/yolo_runs/` | ignorados, sem hashes publicados |
| Manifestos | parcialmente documentado | manifestos de classificação e triagem | inconsistência de modelos solicitados |
| GitHub | completo para código/docs | remoto confirmado em 15/06/2026 | não contém dados nem saídas |
| Google Drive | ausente | nenhum link localizado | definir repositório de dados/artefatos |
| Tag de versão final | ausente | Git sem tags | criar após congelamento |
| Proposta original | ausente | não localizada | anexar ou referenciar documento institucional |
| Bibliografia | ausente | não há seção/repositório de referências | montar e conferir padrão bibliográfico |

O projeto é reproduzível conceitualmente no computador atual, mas ainda não é
reproduzível de forma independente e exata a partir do GitHub.

## 19. Referências existentes e lacunas

### Referências existentes

Não foi localizada bibliografia científica formal em `README.md`, `docs/` ou
nos scripts. O único link externo encontrado é a documentação oficial de
instalação do PyTorch em `README.md`. O DOCX preliminar também não contém uma
bibliografia científica consolidada.

### Referências verificadas recomendadas

1. Medeiros, A. D. de et al. (2020). *Interactive machine learning for soybean
   seed and seedling quality classification*. Scientific Reports, 10, 11267.
   [DOI](https://doi.org/10.1038/s41598-020-68273-y). Sustenta o uso de imagem
   e atributos visuais em qualidade de sementes, com validação independente.
2. Wang, Y.; Song, S. (2024). *Detection of sweet corn seed viability based on
   hyperspectral imaging combined with firefly algorithm optimized deep
   learning*. Frontiers in Plant Science, 15, 1361309.
   [DOI](https://doi.org/10.3389/fpls.2024.1361309). Mostra o ganho potencial
   de informação espectral além do RGB.
3. Zhao, X. et al. (2017). *Early Detection of Aspergillus parasiticus
   Infection in Maize Kernels Using Near-Infrared Hyperspectral Imaging and
   Multivariate Data Analysis*. Applied Sciences, 7(1), 90.
   [DOI](https://doi.org/10.3390/app7010090). Referência diretamente ligada a
   detecção precoce de contaminação fúngica em sementes.
4. Chu, X. et al. (2020). *Classifying Maize Kernels Naturally Infected by
   Fungi Using Near-infrared Hyperspectral Imaging*. Infrared Physics &
   Technology, 105, 103242.
   [DOI](https://doi.org/10.1016/j.infrared.2020.103242). Relevante para
   contaminação natural e validação de sensores NIR.
5. Roberts, D. R. et al. (2017). *Cross-validation strategies for data with
   temporal, spatial, hierarchical, or phylogenetic structure*. Ecography,
   40(8), 913–929. [DOI](https://doi.org/10.1111/ecog.02881). Fundamenta
   validação em blocos/grupos quando existe estrutura dependente.
6. Gulrajani, I.; Lopez-Paz, D. (2021). *In Search of Lost Domain
   Generalization*. ICLR 2021.
   [OpenReview](https://openreview.net/forum?id=lQdXeXDoWtI). Fundamenta
   seleção de modelos e comparação rigorosa em generalização de domínio.
7. Geirhos, R. et al. (2020). *Shortcut learning in deep neural networks*.
   Nature Machine Intelligence, 2, 665–673.
   [DOI](https://doi.org/10.1038/s42256-020-00257-z). Apoia a discussão de
   atalhos por fundo, origem, resolução e tratamento.
8. Geifman, Y.; El-Yaniv, R. (2017). *Selective Classification for Deep Neural
   Networks*. NeurIPS 30.
   [Artigo](https://proceedings.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf).
   Fundamenta triagem com rejeição, cobertura e risco.
9. Chicco, D.; Jurman, G. (2020). *The advantages of the Matthews correlation
   coefficient (MCC) over F1 score and accuracy in binary classification
   evaluation*. BMC Genomics, 21, 6.
   [DOI](https://doi.org/10.1186/s12864-019-6413-7). Justifica não selecionar
   modelos por F1 ou acurácia isoladamente.

### Lacunas teóricas

- estudos específicos de macaúba e contaminação posterior;
- evidência sobre sinais pré-sintomáticos em RGB de sementes;
- modelos que integrem imagem padronizada e variáveis agronômicas causais;
- efeito de lote, câmera, iluminação e tratamento;
- validação em coleta independente;
- comparação direta entre RGB, multiespectral, hiperespectral e NIR;
- desenho de triagem seletiva com utilidade agronômica definida previamente.

As referências sugeridas devem ser formatadas no padrão bibliográfico exigido
pela instituição e conferidas pela orientadora antes do DOCX.

## 20. Informações ainda ausentes

1. arquivo oficial da proposta submetida ao programa;
2. curso, modalidade do programa, edital e eventual agência de fomento;
3. cidade e ano conforme padrão da capa institucional;
4. bibliografia já usada na proposta ou na pesquisa aprofundada;
5. link do Google Drive ou repositório equivalente para dados e saídas;
6. checksums dos dados, tabelas finais e checkpoints;
7. exportação congelada do ambiente com versões exatas;
8. versões de PyTorch, torchvision, Ultralytics, scikit-learn, CUDA e cuDNN;
9. registro de hardware por treinamento;
10. tag `relatorio-final-v1`;
11. tabela específica e reproduzível do T6;
12. fluxograma e gráfico de distribuição das classes;
13. decisão formal sobre o modelo que será chamado de principal;
14. decisão formal sobre a redação da conclusão negativa;
15. versão final sem placeholders do DOCX e da apresentação.

## 21. Estrutura recomendada para o relatório final

1. Capa e folha de rosto.
2. Resumo e palavras-chave.
3. Introdução e relevância da macaúba.
4. Objetivo geral e objetivos específicos.
5. Materiais e métodos.
6. Origem dos dados e desenho experimental.
7. Auditoria, integração e divisão dos dados.
8. Pipeline de visão computacional.
9. Modelos e métricas.
10. Resultados no split original.
11. Validação por tratamento.
12. Análise do T6 e mudança de domínio.
13. Análise de erros e baseline de metadados.
14. Triagem preventiva.
15. Discussão técnico-científica.
16. Justificativa para não implementação do aplicativo.
17. Limitações.
18. Conclusão.
19. Reprodutibilidade e disponibilidade de código/dados.
20. Referências.
21. Apêndices com tabelas completas e parâmetros.

As figuras 1–3 devem entrar em dados/métodos; 4–7 em classificação direta; 8–10
em generalização; 11–12 em erros; 13–14 em triagem.

## 22. Riscos de interpretação que devem ser evitados

1. Não afirmar que a imagem inicial mostra diretamente a infecção.
2. Não chamar o split original de generalização para novos tratamentos.
3. Não chamar a validação por tratamento de coorte externa independente.
4. Não selecionar modelo por acurácia, recall ou F1 isoladamente.
5. Não tratar o baseline de metadados como solução para o aplicativo.
6. Não interpretar SVM ou triagem de recall 1,0 como alto desempenho quando a
   especificidade é zero.
7. Não atribuir causalmente o resultado do T6 a um único elemento visual.
8. Não chamar caixas automáticas de anotações manuais ground truth.
9. Não interpretar rótulos sem imagem como sementes negativas.
10. Não apresentar thresholds calibrados como parâmetros clínicos ou
    agronômicos validados.
11. Não usar `crossfit` como sinônimo de cross-fitting estatístico.
12. Não dizer que o estudo provou inexistência absoluta de sinal visual.
13. Não apresentar a não implementação do aplicativo como abandono: ela é uma
    decisão decorrente do estudo de viabilidade.
14. Não afirmar que o GitHub contém o experimento completo.
15. Não usar os DOCX/PPTX preliminares como fonte de resultados.

## 23. Lista final de decisões que precisam de aprovação humana

| Decisão | Opção tecnicamente recomendada | Responsável sugerido |
|---|---|---|
| Conclusão científica | aprovar a redação refinada da seção 2 | estudante e orientadora |
| Modelo principal | Random Forest como referência de generalização, sem alegação operacional | orientadora |
| Papel da MobileNetV2 | manter como melhor modelo visual no split original e alternativa móvel | estudante e orientadora |
| Aplicativo | registrar formalmente que não foi implementado por falta de segurança científica | orientadora |
| Figuras | aprovar as 14 figuras e decidir se fluxograma/distribuição serão criados depois | estudante e orientadora |
| Bibliografia | aprovar referências, padrão e fontes da proposta original | orientadora |
| Congelamento | criar tag, checksums, pacote de dados/resultados e link de armazenamento | estudante |
| Entregáveis preliminares | decidir se DOCX/PPTX atuais serão descartados ou refeitos após a auditoria | estudante e orientadora |

## Pendências antes da geração do DOCX

| pendência | impacto | arquivo relacionado | ação recomendada | precisa de decisão humana? | prioridade |
|---|---|---|---|---|---|
| aprovar conclusão científica refinada | define toda a discussão e conclusão | este relatório; `docs/relatorio_classificacao_cientifica.md` | validar redação com a orientadora | sim | alta |
| aprovar Random Forest como referência principal | afeta tabelas, resumo e narrativa | `resumo_generalizacao_por_tratamento.csv` | confirmar critério e registrar alternativas | sim | alta |
| formalizar decisão de não desenvolver o aplicativo | evita desalinhamento com o título | `docs/adequacao_escopo_proposta.md` | obter aprovação explícita | sim | alta |
| obter proposta original e dados institucionais faltantes | afeta introdução, objetivos e capa | arquivo não localizado | anexar proposta, curso, edital e fomento | sim | alta |
| montar bibliografia formal | relatório atual não tem base teórica documentada | seção 19 | selecionar e formatar referências verificadas | sim | alta |
| criar link de dados/resultados | GitHub não contém os artefatos ignorados | `.gitignore` | disponibilizar pacote controlado em Drive/repositório institucional | sim | alta |
| congelar ambiente e versões | reprodução exata é impossível | `environment.yml` | exportar versões e registrar CUDA/GPU | não | média |
| reconciliar configuração da validação externa | manifesto e config divergem | `config_validacao_por_tratamento.json` | gerar registro final imutável sem novo treino | não | média |
| documentar divergência do batch YOLO | método pode ser descrito incorretamente | script `15`; `args.yaml` | usar batch 4 no relatório e anotar divergência | não | média |
| produzir tabela dedicada do T6 | melhora rastreabilidade | CSV de métricas por tratamento | filtrar resultados existentes, sem recalibração | não | média |
| decidir fluxograma e distribuição das classes | duas figuras prioritárias não existem | seção 17 | gerar somente após aprovação | sim | média |
| criar tag e checksums | versão final ainda não está congelada | Git e artefatos locais | criar após fechar o DOCX e o pacote científico | sim | alta |
| substituir DOCX/PPTX preliminares | contêm placeholders e não são oficiais | `entrega_final/` | regenerar a partir das decisões aprovadas | sim | alta |
| revisar nomenclatura `crossfit` | evita erro metodológico | arquivos de triagem | explicar em legenda ou renomear em versão futura | não | baixa |
| avaliar resíduos não utilizados | reduz ambiguidade do pacote | `active`, `conda`, `frozen`, `yolo26n.pt` | documentar destino; não apagar sem aprovação | sim | baixa |

## Parecer de prontidão

O projeto está **cientificamente pronto para servir de base ao relatório final,
mas ainda não está pronto para gerar um DOCX definitivo sem ressalvas**. Os
resultados numéricos centrais estão consolidados e a conclusão é estável.
Persistem pendências de decisão científica, identificação institucional,
bibliografia, congelamento de versão e disponibilidade dos artefatos.

As cinco decisões mais importantes antes da geração do DOCX são:

1. aprovar a conclusão científica refinada;
2. aprovar o Random Forest como referência principal, deixando explícito que
   nenhum modelo é operacionalmente adequado;
3. aprovar formalmente a decisão de não implementar o aplicativo;
4. aprovar o conjunto de figuras e a eventual criação do fluxograma e do
   gráfico de classes;
5. definir o congelamento final: bibliografia, proposta original, dados
   institucionais, link dos artefatos, checksums e tag Git.
