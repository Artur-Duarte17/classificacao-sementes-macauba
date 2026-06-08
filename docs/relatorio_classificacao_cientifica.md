# Relatorio cientifico final da classificacao

Gerado em: 2026-06-08T12:48:54

## 1. Objetivo da classificacao

O objetivo da classificacao e estimar, a partir de imagens iniciais e
experimentos associados, o risco de contaminacao posterior em sementes de
macauba. A classe positiva e `contaminada`. A interpretacao cientifica nao deve
ser de deteccao visual direta de infeccao, mas de predicao de risco associada ao
resultado observado posteriormente.

## 2. Amostras e grupos experimentais

A validacao externa foi configurada para 703 amostras e
12 grupos `experimento_tratamento`. Esse grupo combina
`experimento_rotulo` e `tratamento_planilha` normalizados, reduzindo o risco de
que amostras do mesmo contexto experimental aparecam simultaneamente em treino
e teste externo.

Menor grupo externo:

| fold | grupo_externo | n_teste | teste_contaminada | teste_nao_contaminada |
| --- | --- | --- | --- | --- |
| 9.000 | teste_2__t4 | 3.000 | 2.000 | 1.000 |

Maior grupo externo:

| fold | grupo_externo | n_teste | teste_contaminada | teste_nao_contaminada |
| --- | --- | --- | --- | --- |
| 1.000 | micro_ondas__controle | 115.000 | 52.000 | 63.000 |

## 3. Protocolo do split original

O split original usa a divisao treino/validacao/teste consolidada em
`saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv`. Os
modelos e thresholds do split original foram gerados em etapas anteriores; este
script apenas le `comparacao_final_classificacao.csv` e nao recalcula
thresholds.

## 4. Protocolo leave-one-experimento-tratamento-out

Na validacao externa, cada grupo `experimento_tratamento` e deixado de fora uma
vez como teste externo. A validacao interna usa um grupo inteiro do conjunto de
desenvolvimento, escolhido deterministicamente. O split original permanece
apenas como coluna de auditoria.

## 5. Modelos avaliados

Foram consolidados modelos de imagem inteira, YOLO/caixas, ResNet18 com
recortes, Random Forest e SVM com atributos visuais normalizados, MobileNetV2
com recortes, baseline de metadados e baseline sempre-contaminada. O baseline de
metadados e tratado como diagnostico de vies de lote/tratamento, nao como
candidato visual para aplicativo.

## 6. Parametros cientificos principais

```json
{
  "random_forest": {
    "grid_igual_script_23": true,
    "cv": "StratifiedGroupKFold",
    "cv_folds_tentativa": [
      5,
      4,
      3,
      2
    ],
    "n_jobs_grid": 6,
    "n_jobs_random_forest": 1
  },
  "svm_rbf": {
    "grid_igual_script_23": true,
    "pipeline": "SimpleImputer + StandardScaler + SVC RBF",
    "class_weight": "balanced"
  },
  "metadados": {
    "logica_base": "script_26_taxas_suavizadas",
    "alpha_suavizacao": 10.0,
    "papel_experimento": "diagnostico_vies",
    "fit_apply_por_fold": true
  },
  "mobilenetv2": {
    "pesos_pre_treinados": "MobileNet_V2_Weights.DEFAULT",
    "pesos_imagenet_carregados": true,
    "entrada": "224x224",
    "batch_size": 8,
    "num_workers": 4,
    "mixed_precision": true,
    "pin_memory": true,
    "persistent_workers": true,
    "epochs_total": 80,
    "epochs_backbone_congelado": 5,
    "paciencia_early_stopping": 8,
    "learning_rate_classificador": 0.0001,
    "learning_rate_ajuste_fino": 1e-05,
    "weight_decay": 0.0001,
    "blocos_finais_descongelados": 4,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true
  },
  "thresholds": {
    "fixo": 0.5,
    "melhor_f1": "selecionado somente na validacao interna",
    "prioridade_recall": "recall >= 0.95, maior F1, maior especificidade"
  }
}
```

## 7. Resultados do split original

Resultado com maior balanced accuracy entre linhas oficiais/controles do cenario
equilibrado:

metadados_taxas_suavizadas | cenario=teste_threshold_0_50 | features=nao_aplicavel | balanced_accuracy=0.664 | MCC=0.409 | recall=0.938 | especificidade=0.390 | F1=0.808

Tabela resumida do split original:

| modelo | cenario | conjunto_features | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadados_taxas_suavizadas | teste_threshold_0_50 | nao_aplicavel | 0.664 | 0.409 | 0.938 | 0.390 | 0.808 |
| mobilenetv2_recortes | teste_threshold_0_50 | nao_aplicavel | 0.640 | 0.274 | 0.646 | 0.634 | 0.689 |
| random_forest | teste_threshold_0_50 | principal_normalizado | 0.589 | 0.214 | 0.862 | 0.317 | 0.752 |
| recortes_resnet18 | teste_threshold_0_50 | nao_aplicavel | 0.574 | 0.172 | 0.831 | 0.317 | 0.735 |
| svm_rbf | teste_threshold_0_50 | principal_normalizado | 0.500 | 0.000 | 1.000 | 0.000 | 0.760 |
| baseline_sempre_contaminada | teste_baseline_sempre_contaminada | nao_aplicavel | 0.500 | 0.000 | 1.000 | 0.000 | 0.760 |
| baseline_resnet18_imagem_inteira | teste_threshold_0_50 | nao_aplicavel | 0.494 | -0.027 | 0.938 | 0.049 | 0.739 |

## 8. Resultados externos micro e macro

Resultado externo micro com maior balanced accuracy entre linhas consolidadas:

metadados_taxas_suavizadas | cenario=teste_threshold_0_50 | features=nao_aplicavel | balanced_accuracy=0.565 | MCC=0.212 | recall=0.951 | especificidade=0.179 | F1=0.768

Resumo micro:

| modelo | cenario | conjunto_features | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadados_taxas_suavizadas | teste_threshold_0_50 | nao_aplicavel | 0.565 | 0.212 | 0.951 | 0.179 | 0.768 | 12.000 |
| random_forest | teste_threshold_melhor_f1_validacao | principal_normalizado | 0.523 | 0.061 | 0.853 | 0.193 | 0.720 | 12.000 |
| random_forest | teste_threshold_prioridade_recall_validacao | principal_normalizado | 0.523 | 0.061 | 0.853 | 0.193 | 0.720 | 12.000 |
| svm_rbf | teste_threshold_0_50 | principal_normalizado | 0.506 | 0.015 | 0.818 | 0.193 | 0.701 | 12.000 |
| baseline_sempre_contaminada | teste_baseline_sempre_contaminada | nao_aplicavel | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | 12.000 |
| metadados_taxas_suavizadas | teste_threshold_melhor_f1_validacao | nao_aplicavel | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | 12.000 |
| metadados_taxas_suavizadas | teste_threshold_prioridade_recall_validacao | nao_aplicavel | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | 12.000 |
| svm_rbf | teste_threshold_melhor_f1_validacao | principal_normalizado | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | 12.000 |
| svm_rbf | teste_threshold_prioridade_recall_validacao | principal_normalizado | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | 12.000 |
| mobilenetv2_recortes | teste_threshold_melhor_f1_validacao | nao_aplicavel | 0.500 | -0.001 | 0.897 | 0.102 | 0.726 | 12.000 |
| mobilenetv2_recortes | teste_threshold_prioridade_recall_validacao | nao_aplicavel | 0.500 | -0.001 | 0.897 | 0.102 | 0.726 | 12.000 |
| random_forest | teste_threshold_0_50 | principal_normalizado | 0.464 | -0.077 | 0.688 | 0.241 | 0.633 | 12.000 |

Resumo macro:

| modelo | cenario | conjunto_features | balanced_accuracy_media | balanced_accuracy_dp | mcc_media | mcc_dp | recall_contaminada_media | especificidade_nao_contaminada_media | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svm_rbf | teste_threshold_0_50 | principal_normalizado | 0.558 | 0.072 | 0.110 | 0.131 | 0.869 | 0.247 | 12.000 |
| mobilenetv2_recortes | teste_threshold_0_50 | nao_aplicavel | 0.524 | 0.089 | 0.048 | 0.178 | 0.486 | 0.562 | 12.000 |
| random_forest | teste_threshold_melhor_f1_validacao | principal_normalizado | 0.510 | 0.067 | 0.022 | 0.128 | 0.872 | 0.149 | 12.000 |
| random_forest | teste_threshold_prioridade_recall_validacao | principal_normalizado | 0.510 | 0.067 | 0.022 | 0.128 | 0.872 | 0.149 | 12.000 |
| baseline_sempre_contaminada | teste_baseline_sempre_contaminada | nao_aplicavel | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 12.000 |
| metadados_taxas_suavizadas | teste_threshold_0_50 | nao_aplicavel | 0.500 | 0.000 | 0.000 | 0.000 | 0.708 | 0.292 | 12.000 |
| metadados_taxas_suavizadas | teste_threshold_melhor_f1_validacao | nao_aplicavel | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 12.000 |
| metadados_taxas_suavizadas | teste_threshold_prioridade_recall_validacao | nao_aplicavel | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 12.000 |
| svm_rbf | teste_threshold_melhor_f1_validacao | principal_normalizado | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 12.000 |
| svm_rbf | teste_threshold_prioridade_recall_validacao | principal_normalizado | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 12.000 |
| mobilenetv2_recortes | teste_threshold_melhor_f1_validacao | nao_aplicavel | 0.494 | 0.081 | -0.014 | 0.163 | 0.837 | 0.150 | 12.000 |
| mobilenetv2_recortes | teste_threshold_prioridade_recall_validacao | nao_aplicavel | 0.494 | 0.081 | -0.014 | 0.163 | 0.837 | 0.150 | 12.000 |

## 9. Comparacao com baseline sempre-contaminada

O baseline sempre-contaminada e um controle obrigatorio: ele tende a maximizar
recall da classe contaminada ao custo de especificidade nula ou muito baixa.
Resultados com F1 alto devem ser interpretados contra esse controle.

Split original:

| cenario | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada |
| --- | --- | --- | --- | --- | --- |
| teste_baseline_sempre_contaminada | 0.500 | 0.000 | 1.000 | 0.000 | 0.760 |

Validacao externa:

| agregacao | cenario | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada | balanced_accuracy_media | mcc_media |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| micro | teste_baseline_sempre_contaminada | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | NA | NA |
| macro | teste_baseline_sempre_contaminada | NA | NA | NA | NA | NA | 0.500 | 0.000 |

## 10. Diagnostico de vies de lote/tratamento

O baseline de metadados usa origem, tratamento, pasta e campos derivados. Ele
serve para diagnosticar vies de lote/tratamento e nao deve ser tratado como
modelo visual candidato ao aplicativo.

Split original metadados:

| cenario | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada |
| --- | --- | --- | --- | --- | --- |
| teste_threshold_0_50 | 0.664 | 0.409 | 0.938 | 0.390 | 0.808 |
| teste_threshold_melhor_f1_validacao | 0.664 | 0.409 | 0.938 | 0.390 | 0.808 |
| teste_threshold_prioridade_recall_validacao | 0.570 | 0.245 | 0.969 | 0.171 | 0.778 |

Validacao externa metadados:

| agregacao | cenario | balanced_accuracy | mcc | recall_contaminada | especificidade_nao_contaminada | f1_contaminada | balanced_accuracy_media | mcc_media |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| micro | teste_threshold_0_50 | 0.565 | 0.212 | 0.951 | 0.179 | 0.768 | NA | NA |
| macro | teste_threshold_0_50 | NA | NA | NA | NA | NA | 0.500 | 0.000 |
| micro | teste_threshold_melhor_f1_validacao | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | NA | NA |
| macro | teste_threshold_melhor_f1_validacao | NA | NA | NA | NA | NA | 0.500 | 0.000 |
| micro | teste_threshold_prioridade_recall_validacao | 0.500 | 0.000 | 1.000 | 0.000 | 0.758 | NA | NA |
| macro | teste_threshold_prioridade_recall_validacao | NA | NA | NA | NA | NA | 0.500 | 0.000 |

Resumo textual do script 27:

```text
Resumo da comparacao final de classificacao
================================================

Melhor modelo visual equilibrado:
mobilenetv2_recortes | cenario=teste_threshold_0_50 | features=nao_aplicavel | recall=0.646 | especificidade=0.634 | F1=0.689 | balanced_accuracy=0.640 | MCC=0.274

Melhor modelo visual com prioridade de recall:
svm_rbf | cenario=teste_threshold_prioridade_recall_validacao | features=principal_normalizado | recall=1.000 | especificidade=0.000 | F1=0.760 | balanced_accuracy=0.500 | MCC=0.000

Comparacao com baseline sempre contaminada:
baseline_sempre_contaminada | cenario=teste_baseline_sempre_contaminada | features=nao_aplicavel | recall=1.000 | especificidade=0.000 | F1=0.760 | balanced_accuracy=0.500 | MCC=0.000
O melhor modelo visual de prioridade de recall ganha 0.000 em especificidade e 0.000 em balanced accuracy contra esse controle.

Comparacao com baseline de metadados:
metadados_taxas_suavizadas | cenario=teste_threshold_0_50 | features=nao_aplicavel | recall=0.938 | especificidade=0.390 | F1=0.808 | balanced_accuracy=0.664 | MCC=0.409
Diferenca do melhor visual equilibrado contra metadados: F1=-0.119, balanced_accuracy=-0.024, MCC=-0.135.

Modelos com recall alto e especificidade quase nula:
svm_rbf | cenario=teste_threshold_0_50 | features=principal_normalizado | recall=1.000 | especificidade=0.000 | F1=0.760 | balanced_accuracy=0.500 | MCC=0.000
svm_rbf | cenario=teste_threshold_melhor_f1_validacao | features=principal_normalizado | recall=1.000 | especificidade=0.000 | F1=0.760 | balanced_accuracy=0.500 | MCC=0.000
svm_rbf | cenario=teste_threshold_prioridade_recall_validacao | features=principal_normalizado | recall=1.000 | especificidade=0.000 | F1=0.760 | balanced_accuracy=0.500 | MCC=0.000
baseline_resnet18_imagem_inteira | cenario=teste_threshold_melhor_f1_validacao | features=nao_aplicavel | recall=0.985 | especificidade=0.000 | F1=0.753 | balanced_accuracy=0.492 | MCC=-0.078
random_forest | cenario=teste_threshold_melhor_f1_validacao | features=principal_normalizado | recall=0.969 | especificidade=0.073 | F1=0.759 | balanced_accuracy=0.521 | MCC=0.097
random_forest | cenario=teste_threshold_prioridade_recall_validacao | features=principal_normalizado | recall=0.969 | especificidade=0.073 | F1=0.759 | balanced_accuracy=0.521 | MCC=0.097
recortes_resnet18 | cenario=teste_threshold_melhor_f1_validacao | features=nao_aplicavel | recall=0.969 | especificidade=0.049 | F1=0.754 | balanced_accuracy=0.509 | MCC=0.046
recortes_resnet18 | cenario=teste_threshold_prioridade_recall_validacao | features=nao_aplicavel | recall=0.969 | especificidade=0.049 | F1=0.754 | balanced_accuracy=0.509 | MCC=0.046

Notas de interpretacao:
- O baseline de metadados e diagnostico de vies de lote/tratamento, nao candidato visual para aplicativo.
- O Random Forest com sensibilidade_todos_atributos e analise de sensibilidade, nao resultado oficial.
- Resultados com recall alto e especificidade muito baixa podem estar proximos do controle sempre contaminada.
- Esta comparacao nao prova generalizacao definitiva; o script 28 deve validar por tratamento/lote.
```

## 11. Limitacoes

As conclusoes sao limitadas pelo tamanho da base, pelo numero de grupos
experimentais e por possiveis diferencas tecnicas entre lotes, tratamentos,
origens e padroes de imagem. Grupos pequenos reduzem a estabilidade das metricas
por tratamento e ampliam incerteza na leitura macro.

Grupos pequenos no diagnostico dos folds:

| fold | grupo_externo | n_teste | teste_contaminada | teste_nao_contaminada |
| --- | --- | --- | --- | --- |
| 9.000 | teste_2__t4 | 3.000 | 2.000 | 1.000 |

## 12. Conclusao sobre viabilidade da classificacao

Nao se deve declarar vencedor apenas por F1. A leitura prioritaria deve combinar
balanced accuracy, MCC, recall da contaminada e especificidade da nao
contaminada. Se a validacao externa apresentar queda relevante em balanced
accuracy ou MCC frente ao split original, isso indica fragilidade de
generalizacao e reforca que o problema ainda nao esta pronto para classificacao
direta automatica.

## 13. Justificativa para avancar para triagem preventiva

A classificacao direta exige boa sensibilidade sem destruir a especificidade. O
historico dos experimentos mostra que recall alto pode ser obtido por regras
conservadoras proximas ao baseline sempre-contaminada. Portanto, a etapa
operacional mais defensavel e triagem preventiva: separar alto risco, revisar
casos incertos e evitar liberacao automatica de baixo risco sem validacao
adicional por lote/tratamento.

## Figuras

![metricas_split_original](saidas/tabelas/07_classificacao_final/relatorio/figuras/metricas_split_original.png)
![metricas_validacao_externa_micro](saidas/tabelas/07_classificacao_final/relatorio/figuras/metricas_validacao_externa_micro.png)
![variacao_entre_tratamentos](saidas/tabelas/07_classificacao_final/relatorio/figuras/variacao_entre_tratamentos.png)
![comparacao_split_original_vs_validacao_externa](saidas/tabelas/07_classificacao_final/relatorio/figuras/comparacao_split_original_vs_validacao_externa.png)

## Arquivos derivados deste relatorio

- `saidas\tabelas\07_classificacao_final\relatorio\tabela_resultados_split_original.csv`
- `saidas\tabelas\07_classificacao_final\relatorio\tabela_resultados_validacao_externa.csv`
- `saidas\tabelas\07_classificacao_final\relatorio\tabela_desempenho_por_tratamento.csv`
- `saidas\tabelas\07_classificacao_final\relatorio\manifesto_experimento_final.json`
