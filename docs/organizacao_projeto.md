# Organizacao do projeto

Data da revisao: 04/06/2026.

Este documento descreve a organizacao atual do projeto depois da consolidacao em uma unica pipeline.

## Direcao atual

O projeto nao esta mais dividido em fases. A estrutura agora separa responsabilidades por tipo de tarefa:

- preparacao de dados;
- modelos de imagem inteira;
- caixas/YOLO;
- modelos com recortes;
- triagem e analise integrada;
- baseline de metadados.

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
| `06_modelos\metadados\` | baseline sem pixels |
| `06_modelos\comparacao\` | comparacoes consolidadas |
| `07_triagem\` | tabela integrada e calibracao operacional |

Pastas antigas como `07_fase2_triagem\` podem existir localmente por compatibilidade com execucoes anteriores. Os scripts novos escrevem em `07_triagem\` e alguns leem a pasta antiga apenas como fallback.

## Duplicacoes removidas

Foi removido o baseline de metadados duplicado que dependia de `scikit-learn` e salvava resultados em uma pasta separada.

O baseline de metadados mantido e:

```text
scripts\triagem\27_baseline_metadados.py
```

Razao: ele usa o mesmo split dos modelos de imagem, gera metricas comparaveis em `06_modelos`, salva a comparacao consolidada e roda sem depender de `scikit-learn` no runtime empacotado usado neste ambiente.

## Scripts integrados

Use apenas a pasta:

```text
scripts\triagem\
```

Ela substitui a antiga organizacao separada e concentra tabela integrada, triagem, calibracao, comparacao de scores e baseline de metadados.

## Artefatos pesados

Pastas derivadas como estas podem ser arquivadas fora do projeto quando nao forem necessarias para auditoria imediata:

```text
saidas\dataset_binario\
saidas\dataset_recortado\
saidas\yolo_dataset\
saidas\conferencia_caixas\
saidas\conferencia_yolo\
saidas\conferencia_recortes\
saidas\amostras_conferencia\
```

Arquivar e preferivel a apagar, porque algumas conferencias visuais podem ser uteis para rastrear erros.

## Comando principal de retomada

Para retomar a analise integrada com os artefatos ja gerados:

```powershell
python scripts\triagem\22_criar_tabela_integrada.py
python scripts\triagem\23_analisar_triagem.py
python scripts\triagem\27_baseline_metadados.py
```

Para recalibrar a triagem com predicoes em todos os splits, rode tambem:

```powershell
python scripts\triagem\24_gerar_predicoes_todos_splits.py
python scripts\triagem\25_calibrar_thresholds_triagem.py
python scripts\triagem\26_comparar_scores_triagem.py
```
