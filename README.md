# classificacao-sementes-macauba

Projeto de iniciacao cientifica/prototipo rapido para estimar risco de contaminacao em sementes de macauba a partir de imagens iniciais, metadados do experimento e resultados das planilhas.

A classe positiva do problema continua sendo `contaminada`. As metricas prioritarias sao:

- recall/sensibilidade da classe `contaminada`;
- especificidade da classe `nao_contaminada`;
- F1 da classe `contaminada`;
- seguranca operacional da triagem, principalmente evitar liberar sementes contaminadas.

## Contexto cientifico

As imagens foram tiradas no comeco dos tratamentos. A contaminacao foi observada depois e registrada nas planilhas.

Portanto, o modelo nao deve ser descrito como uma IA que enxerga infeccao diretamente na imagem inicial. O enquadramento correto e: modelo preditivo que procura sinais associados a contaminacao registrada posteriormente.

O projeto agora fica organizado como uma unica pipeline. A triagem e a analise de metadados fazem parte do mesmo fluxo, nao de uma fase separada.

## Estrutura

```text
C:\Projetos\sementes_ia
  dados_originais\
    imagens\
    planilhas\
  scripts\
    preparacao\
    baseline\
    caixas_yolo\
    recortes\
    triagem\
  docs\
  saidas\
    amostras_conferencia\
    conferencia_caixas\
    conferencia_recortes\
    conferencia_yolo\
    dataset_binario\
    dataset_recortado\
    tabelas\
    figuras\
    modelos\
    yolo_dataset\
    yolo_runs\
```

`dados_originais/` e `saidas/` nao devem ir para o GitHub, porque contem imagens, tabelas derivadas, modelos e arquivos locais/pesados.

## Pacotes de scripts

| Pacote | Conteudo |
|---|---|
| `scripts\preparacao\` | inventario, leitura de planilhas, rotulos, tabela mestre e dataset binario |
| `scripts\baseline\` | treino e avaliacao da ResNet18 com imagem inteira |
| `scripts\caixas_yolo\` | caixas automaticas, ajustes, dataset YOLO, treino YOLO e erros |
| `scripts\recortes\` | classificacao com recortes, comparacoes, erros e baseline de metadados |
| `scripts\triagem\` | tabela integrada, triagem operacional, calibracao e comparacao de scores |

## Pacotes de tabelas

| Pasta | Conteudo |
|---|---|
| `01_inventario\` | inventario das imagens e problemas de leitura/nome |
| `02_planilhas_rotulos\` | leitura das planilhas, previas, rotulos e duplicatas de rotulo |
| `03_tabela_mestre\` | cruzamento imagem + rotulo e tabelas treinaveis |
| `04_dataset_split\` | relatorio do dataset binario e divisao treino/validacao/teste |
| `05_caixas_yolo\` | caixas automaticas, ajustes manuais e relatorio do dataset YOLO |
| `06_modelos\baseline\` | historico, metricas, thresholds e predicoes do baseline de imagem inteira |
| `06_modelos\yolo\` | metricas, thresholds, predicoes e resumo por origem do YOLO |
| `06_modelos\recortes\` | historico, metricas, thresholds, predicoes e erros dos recortes |
| `06_modelos\classicos\` | atributos visuais extraidos dos recortes para modelos classicos |
| `06_modelos\mobilenetv2\` | historico, metricas, thresholds e predicoes da MobileNetV2 nos recortes |
| `06_modelos\metadados\` | baseline usando apenas origem, tratamento, pasta e campos derivados |
| `06_modelos\comparacao\` | comparacoes consolidadas dos modelos |
| `07_classificacao_final\` | comparacao final, ranking e conclusao cientifica da classificacao |
| `08_triagem\` | tabela integrada, predicoes em todos os splits, calibracao e conclusoes operacionais |

## Ambiente

Ative o ambiente conda:

```powershell
conda activate sementes_ia
```

Instale/atualize dependencias:

```powershell
conda env update -f environment.yml
```

Para treino com GPU, instale PyTorch com CUDA seguindo o seletor oficial:

https://pytorch.org/get-started/locally/

Confira se a GPU foi detectada:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Ordem operacional

Rode apenas os blocos necessarios. Os nomes numericos indicam a ordem recomendada.

### 1. Preparar dados

```powershell
python scripts\preparacao\00_inventario_imagens.py
python scripts\preparacao\01_ler_planilhas_rotulos.py
python scripts\preparacao\02_criar_rotulos_planilhas.py
python scripts\preparacao\03_criar_tabela_mestre.py
python scripts\preparacao\04_criar_dataset_binario.py
python scripts\preparacao\05_conferir_amostras_dataset.py
```

Saidas principais:

- `saidas\tabelas\03_tabela_mestre\tabela_mestre.csv`
- `saidas\tabelas\03_tabela_mestre\tabela_mestre_treinavel.csv`
- `saidas\tabelas\04_dataset_split\divisao_treino_validacao_teste.csv`

### 2. Baseline com imagem inteira

```powershell
python scripts\baseline\06_treinar_baseline.py
python scripts\baseline\07_avaliar_modelo.py
```

Saidas principais:

- `saidas\modelos\baseline_resnet18_melhor.pt`
- `saidas\tabelas\06_modelos\baseline\metricas_baseline_resnet18_teste.csv`
- `saidas\tabelas\06_modelos\baseline\predicoes_baseline_resnet18_teste.csv`

### 3. Caixas e YOLO

```powershell
python scripts\caixas_yolo\08_gerar_caixas_microondas.py
python scripts\caixas_yolo\09_gerar_caixas_piloto_teste2.py
python scripts\caixas_yolo\10_juntar_caixas_automaticas.py
python scripts\caixas_yolo\13_conferir_caixas_automaticas.py
```

Se precisar ajustar caixas manualmente:

```powershell
python scripts\caixas_yolo\11_marcar_ajustes_manuais_caixas.py --filtro TESTE_2__T6__
python scripts\caixas_yolo\12_aplicar_ajustes_manuais_caixas.py
python scripts\caixas_yolo\13_conferir_caixas_automaticas.py
```

Depois:

```powershell
python scripts\caixas_yolo\14_criar_dataset_yolo.py
python scripts\caixas_yolo\15_treinar_yolo.py
python scripts\caixas_yolo\16_avaliar_yolo.py
python scripts\caixas_yolo\17_conferir_erros_yolo.py
```

### 4. Fechamento da classificacao

```powershell
python scripts\recortes\18_treinar_recortes_resnet18.py
python scripts\recortes\19_avaliar_recortes_resnet18.py
python scripts\recortes\20_comparar_resultados_modelos.py
python scripts\recortes\21_conferir_erros_recortes.py
python scripts\recortes\22_extrair_atributos_visuais_recortes.py
python scripts\recortes\23_treinar_avaliar_classicos_recortes.py
python scripts\recortes\24_treinar_mobilenetv2_recortes.py
python scripts\recortes\25_avaliar_mobilenetv2_recortes.py
python scripts\recortes\26_baseline_metadados_classificacao.py
```

Objetivo deste bloco:

- testar se remover fundo, regua e bancada melhora o classificador;
- comparar contra o baseline de imagem inteira e YOLO;
- extrair atributos visuais interpretaveis dos recortes;
- treinar Random Forest e SVM RBF com CV estratificada no treino;
- treinar e avaliar MobileNetV2 com entrada 224x224 e pesos ImageNet;
- testar se origem, tratamento, pasta e outros campos derivados explicam a predicao sem usar pixels;
- medir recall, especificidade e F1.

Saidas principais:

- `saidas\tabelas\06_modelos\metadados\metricas_metadados_teste.csv`
- `saidas\tabelas\06_modelos\comparacao\comparacao_metadados_vs_modelos_teste.csv`
- `saidas\tabelas\06_modelos\classicos\atributos_visuais_recortes.csv`
- `saidas\tabelas\06_modelos\classicos\resumo_atributos_visuais_recortes.csv`
- `saidas\tabelas\06_modelos\classicos\metricas_classicos_teste.csv`
- `saidas\tabelas\06_modelos\classicos\predicoes_classicos_teste.csv`
- `saidas\tabelas\06_modelos\classicos\curva_threshold_classicos_validacao.csv`
- `saidas\tabelas\06_modelos\classicos\melhores_parametros_classicos.csv`
- `saidas\tabelas\06_modelos\classicos\importancia_random_forest.csv`
- `saidas\modelos\mobilenetv2_recortes_melhor.pt`
- `saidas\tabelas\06_modelos\mobilenetv2\historico_treino_mobilenetv2_recortes.csv`
- `saidas\tabelas\06_modelos\mobilenetv2\metricas_mobilenetv2_recortes_teste.csv`
- `saidas\tabelas\06_modelos\mobilenetv2\predicoes_mobilenetv2_recortes_teste.csv`

O script 23 avalia dois conjuntos: `principal_normalizado`, sem medidas absolutas de resolucao/posicao nem textura na resolucao original, e `sensibilidade_todos_atributos`, com todas as features visuais para medir possivel ganho artificial.

A MobileNetV2 usa `batch_size=8`, `num_workers=4`, mixed precision em CUDA, `pin_memory=True`, `persistent_workers=True`, entrada `224x224` e pesos ImageNet quando disponiveis.

Os scripts `27-29` serao adicionados nas proximas etapas para comparacao final, validacao por tratamento e relatorio cientifico.

### 5. Triagem operacional

```powershell
python scripts\triagem\30_criar_tabela_integrada.py
python scripts\triagem\31_analisar_triagem.py
python scripts\triagem\32_gerar_predicoes_todos_splits.py
python scripts\triagem\33_calibrar_thresholds_triagem.py
python scripts\triagem\34_comparar_scores_triagem.py
```

Objetivo deste bloco:

- consolidar rotulos, metadados, split e predicoes em `tabela_integrada.csv`;
- avaliar regras de `alto_risco`, `baixo_risco` e `incerto`;
- calibrar thresholds usando validacao e avaliar no teste;
- comparar scores alternativos para triagem conservadora.

Saidas principais:

- `saidas\tabelas\08_triagem\tabela_integrada.csv`
- `saidas\tabelas\08_triagem\predicoes_todos_splits.csv`
- `saidas\tabelas\08_triagem\thresholds_triagem_recomendados.csv`
- `saidas\tabelas\08_triagem\score_triagem_recomendado.csv`

## Decisao operacional atual

Com os resultados atuais, a triagem deve ser tratada como alerta conservador:

```text
alto_risco -> separar
incerto -> revisar manualmente
baixo_risco -> nao usar para liberacao automatica sem nova validacao
```

O baseline de metadados indicou sinal forte de lote/tratamento. Isso deve ser considerado na interpretacao dos modelos de imagem e em novas divisoes experimentais.

## GitHub

Este repositorio deve versionar:

- `scripts/`
- `docs/`
- `README.md`
- `.gitignore`
- `environment.yml`

Nao versionar:

- `dados_originais/`
- `saidas/`
- imagens;
- modelos `.pt`;
- arquivos `.zip`;
- documentos pessoais ou relatorios em `.docx`/`.pdf`.
