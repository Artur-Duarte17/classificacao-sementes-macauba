# Organizacao do projeto

Data da revisao: 04/06/2026.

Este documento descreve a organizacao atual do projeto depois da consolidacao em uma unica pipeline.

## Direcao atual

O projeto nao esta mais dividido em fases. A estrutura agora separa responsabilidades por tipo de tarefa:

- preparacao de dados;
- modelos de imagem inteira;
- caixas/YOLO;
- modelos com recortes;
- baseline de metadados e fechamento de classificacao;
- triagem operacional.

## Estrutura versionada

```text
README.md
environment.yml
.gitignore
docs\
scripts\
```

Pacotes de scripts:

```text
scripts\
  preparacao\
  baseline\
  caixas_yolo\
  recortes\
  triagem\
```

## Dados locais

Nao versionar:

```text
dados_originais\
saidas\
*.pt
*.zip
*.docx
*.pdf
```

Motivo:

- `dados_originais\` contem imagens e planilhas fonte;
- `saidas\` contem datasets derivados, modelos, figuras e CSVs gerados;
- modelos e imagens podem ser grandes e devem ser regeneraveis pelos scripts.

## Tabelas derivadas

Organizacao recomendada dentro de `saidas\tabelas\`:

| Pasta | Funcao |
|---|---|
| `01_inventario\` | inventario e diagnostico das imagens |
| `02_planilhas_rotulos\` | leitura das planilhas e rotulos consolidados |
| `03_tabela_mestre\` | cruzamento imagem + rotulo |
| `04_dataset_split\` | copia do dataset binario e splits |
| `05_caixas_yolo\` | pseudo-caixas, ajustes e relatorio YOLO |
| `06_modelos\baseline\` | metricas/predicoes de imagem inteira |
| `06_modelos\yolo\` | metricas/predicoes YOLO |
| `06_modelos\recortes\` | metricas/predicoes de recortes |
| `06_modelos\classicos\` | atributos visuais, Random Forest e SVM dos recortes |
| `06_modelos\mobilenetv2\` | historico, metricas e predicoes da MobileNetV2 |
| `06_modelos\metadados\` | baseline sem pixels |
| `06_modelos\comparacao\` | comparacoes consolidadas |
| `07_classificacao_final\` | comparacao final, rankings, validacao por tratamento e resumo cientifico |
| `08_triagem\` | tabela integrada e calibracao operacional |

Pastas antigas como `07_triagem\` podem existir localmente por compatibilidade com execucoes anteriores. Os scripts novos escrevem em `08_triagem\` e alguns leem `07_triagem\` apenas como fallback.

## Duplicacoes removidas

Foi removido o baseline de metadados duplicado que dependia de `scikit-learn` e salvava resultados em uma pasta separada.

O baseline de metadados mantido e:

```text
scripts\recortes\26_baseline_metadados_classificacao.py
```

Razao: ele usa o mesmo split dos modelos de imagem, gera metricas comparaveis em `06_modelos`, salva a comparacao consolidada e roda sem depender de `scikit-learn` no runtime empacotado usado neste ambiente.

## Scripts de classificacao

O fechamento de classificacao fica em:

```text
scripts\recortes\
```

Ela concentra treino/avaliacao dos recortes, comparacoes dos modelos de imagem, auditoria de erros e baseline de metadados sem pixels.

O script final de comparacao da classificacao e:

```text
scripts\recortes\27_comparar_classificacao_final.py
```

Ele consolida metricas de baseline, YOLO, recortes, modelos classicos, MobileNetV2 e metadados em `saidas\tabelas\07_classificacao_final\`, mantendo o baseline de metadados como diagnostico de vies e separando analises de sensibilidade dos resultados oficiais.

A validacao externa por tratamento/lote fica em:

```text
scripts\recortes\28_validacao_por_tratamento_classificacao.py
```

Ela usa `experimento_tratamento`, formado por `experimento_rotulo + tratamento_planilha` normalizados, como grupo externo. Nesse protocolo, o split original nao define treino/validacao/teste; ele fica apenas como coluna de auditoria. As saidas sao escritas em `saidas\tabelas\07_classificacao_final\validacao_tratamento\`.

O arquivo `28_validacao_por_tratamento_classificacao.py` e mantido como entrypoint para preservar os comandos. A implementacao fica em:

```text
scripts\recortes\validacao_tratamento\
```

Principais responsabilidades:

- `config.py`: caminhos, constantes cientificas e contexto dos modelos;
- `dados.py`: juncoes, atributos visuais e metadados brutos;
- `folds.py`: construcao e diagnostico dos grupos externos;
- `classicos.py`, `metadados.py`, `mobilenet.py` e `controles.py`: modelos;
- `metricas.py` e `thresholds.py`: calculos de desempenho e cenarios;
- `persistencia.py`: escrita atomica, retomada e config;
- `agregacao.py`: micro/macro, comparacao com split original e auditoria;
- `runner.py`: CLI e orquestracao.

## Scripts de triagem

Use a pasta:

```text
scripts\triagem\
```

Ela concentra tabela integrada, triagem operacional, calibracao e comparacao de scores. A numeracao da triagem comeca em `30`, depois do fechamento de classificacao.

## Artefatos pesados

Os artefatos derivados ficam unificados dentro de `saidas\`. Nao ha mais pasta separada de arquivo para a etapa antiga.

```text
saidas\amostras_conferencia\
saidas\conferencia_caixas\
saidas\conferencia_recortes\
saidas\conferencia_yolo\
saidas\dataset_binario\
saidas\dataset_recortado\
saidas\yolo_dataset\
saidas\yolo_runs\
```

Essas pastas continuam fora do Git por causa do `.gitignore`, mas ficam no projeto principal para facilitar auditoria, reexecucao e retomada da pipeline.

## Comando principal de retomada

Para retomar a analise integrada com os artefatos ja gerados:

```powershell
python scripts\recortes\22_extrair_atributos_visuais_recortes.py
python scripts\recortes\23_treinar_avaliar_classicos_recortes.py
python scripts\recortes\24_treinar_mobilenetv2_recortes.py
python scripts\recortes\25_avaliar_mobilenetv2_recortes.py
python scripts\recortes\26_baseline_metadados_classificacao.py
python scripts\recortes\27_comparar_classificacao_final.py
```

Para validar generalizacao por tratamento, rode primeiro:

```powershell
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --preflight
```

Depois, conforme o tempo disponivel no conda:

```powershell
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --modelos random_forest svm_rbf metadados
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --modelos mobilenetv2 --retomar
```

Para iniciar a triagem operacional depois do fechamento da classificacao:

```powershell
python scripts\triagem\30_criar_tabela_integrada.py
python scripts\triagem\31_analisar_triagem.py
```

Para recalibrar a triagem com predicoes em todos os splits, rode tambem:

```powershell
python scripts\triagem\32_gerar_predicoes_todos_splits.py
python scripts\triagem\33_calibrar_thresholds_triagem.py
python scripts\triagem\34_comparar_scores_triagem.py
```
