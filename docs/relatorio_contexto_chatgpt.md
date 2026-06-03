# Relatorio de contexto do projeto sementes_ia

Este documento resume o projeto `classificacao-sementes-macauba` ate a etapa atual. Ele foi escrito para ser enviado a uma nova conversa do ChatGPT junto com alguns arquivos, permitindo que a nova conversa entenda o objetivo, os dados, os scripts, os resultados e a conclusao tecnica/cientifica.

Ultima verificacao deste relatorio: 02/06/2026, apos executar `python scripts\recortes\21_conferir_erros_recortes.py`. Os resultados de falsos positivos/falsos negativos e o resumo por origem dos recortes estao atualizados com essa execucao.

## 1. Objetivo do projeto

O projeto busca avaliar se imagens iniciais de sementes de macauba conseguem prever a classe final registrada nas planilhas:

- `contaminada`
- `nao_contaminada`

A classe positiva e `contaminada`.

A prioridade cientifica foi maximizar a sensibilidade/recall da classe `contaminada`, porque o erro mais perigoso e classificar uma semente contaminada como `nao_contaminada`.

Porem, tambem foi acompanhada a especificidade da classe `nao_contaminada`, porque muitos falsos positivos tornam o modelo pouco util na pratica.

## 2. Enquadramento cientifico correto

As fotos foram tiradas no inicio dos tratamentos. A contaminacao foi observada depois e registrada nas planilhas.

Portanto, o modelo nao deve ser descrito como uma IA que "enxerga contaminacao" diretamente na imagem inicial.

O enquadramento correto e:

> modelo preditivo que procura padroes visuais iniciais associados a contaminacao registrada posteriormente.

Pelos resultados obtidos, ha evidencia de que o sinal visual nas imagens iniciais e fraco ou muito parecido entre as classes.

## 3. Ambiente e maquina

Projeto local:

```text
C:\Projetos\sementes_ia
```

Ambiente conda usado:

```text
C:\Projeto_de_Pesquisa\laboratorio\sementes_ia
```

Comandos usados:

```powershell
conda activate sementes_ia
cd C:\Projetos\sementes_ia
```

Maquina informada:

- Windows 11 Home
- CPU Intel Core i7-12650H
- RAM 24 GB
- GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- VRAM dedicada 4 GB GDDR6

O treino com PyTorch detectou CUDA corretamente:

```text
Dispositivo usado: cuda
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

## 4. Estrutura geral dos dados

O dataset binario foi gerado em:

```text
saidas\dataset_binario\
  contaminada\
  nao_contaminada\
```

Total de imagens processadas no dataset:

- `contaminada`: 429
- `nao_contaminada`: 274
- total: 703

A divisao treino/validacao/teste foi reutilizada em todos os experimentos para comparacao justa:

| split | contaminada | nao_contaminada |
|---|---:|---:|
| treino | 299 | 192 |
| validacao | 65 | 41 |
| teste | 65 | 41 |

O conjunto de teste possui 106 imagens.

## 5. Ordem atual dos scripts

### Preparacao dos dados

```powershell
python scripts\preparacao\00_inventario_imagens.py
python scripts\preparacao\01_ler_planilhas_rotulos.py
python scripts\preparacao\02_criar_rotulos_planilhas.py
python scripts\preparacao\03_criar_tabela_mestre.py
python scripts\preparacao\04_criar_dataset_binario.py
python scripts\preparacao\05_conferir_amostras_dataset.py
```

### Baseline com imagem inteira

```powershell
python scripts\baseline\06_treinar_baseline.py
python scripts\baseline\07_avaliar_modelo.py
```

### Caixas, recortes e YOLO

```powershell
python scripts\caixas_yolo\08_gerar_caixas_microondas.py
python scripts\caixas_yolo\09_gerar_caixas_piloto_teste2.py
python scripts\caixas_yolo\10_juntar_caixas_automaticas.py
python scripts\caixas_yolo\11_marcar_ajustes_manuais_caixas.py --filtro TESTE_2__T6__
python scripts\caixas_yolo\12_aplicar_ajustes_manuais_caixas.py
python scripts\caixas_yolo\13_conferir_caixas_automaticas.py
python scripts\caixas_yolo\14_criar_dataset_yolo.py
python scripts\caixas_yolo\15_treinar_yolo.py
python scripts\caixas_yolo\16_avaliar_yolo.py
python scripts\caixas_yolo\17_conferir_erros_yolo.py
```

### Classificador com recortes

```powershell
python scripts\recortes\18_treinar_recortes_resnet18.py
python scripts\recortes\19_avaliar_recortes_resnet18.py
python scripts\recortes\20_comparar_resultados_modelos.py
python scripts\recortes\21_conferir_erros_recortes.py
```

## 6. Experimentos realizados

### 6.1 Baseline ResNet18 com imagem inteira

Objetivo:

- usar a imagem completa sem segmentacao;
- ter uma referencia inicial simples;
- salvar o melhor modelo com foco em recall da classe `contaminada`.

Arquivos principais:

```text
scripts\baseline\06_treinar_baseline.py
scripts\baseline\07_avaliar_modelo.py
saidas\modelos\baseline_resnet18_melhor.pt
saidas\tabelas\06_modelos\baseline\metricas_baseline_resnet18_teste.csv
```

Resultado no teste:

| cenario | threshold | acuracia | precisao contaminada | recall/sensibilidade contaminada | especificidade nao contaminada | F1 contaminada | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.50 | 0.50 | 0.5943 | 0.6100 | 0.9385 | 0.0488 | 0.7394 | 2 | 39 | 4 | 61 |
| melhor F1 validacao | 0.25 | 0.6038 | 0.6095 | 0.9846 | 0.0000 | 0.7529 | 0 | 41 | 1 | 64 |

Interpretacao:

- O baseline consegue recall muito alto.
- No threshold 0.25, perde apenas 1 contaminada no teste.
- Mas a especificidade vira 0: todas as 41 imagens `nao_contaminada` do teste sao marcadas como `contaminada`.
- Isso indica um modelo muito sensivel, mas pouco especifico.

### 6.2 YOLO com pseudo-caixas automaticas

Objetivo:

- localizar a semente com caixas automaticas por OpenCV;
- converter as caixas para formato YOLO;
- testar se um detector/classificador por caixa melhoraria o resultado.

Importante:

- As caixas sao pseudo-rotulos, nao anotacoes manuais.
- O YOLO usou imagens originais e arquivos de label com caixas.
- As imagens em `saidas\dataset_recortado\` foram geradas para conferencia e para o experimento posterior de classificador com recortes.

Arquivos principais:

```text
scripts\caixas_yolo\08_gerar_caixas_microondas.py
scripts\caixas_yolo\09_gerar_caixas_piloto_teste2.py
scripts\caixas_yolo\10_juntar_caixas_automaticas.py
scripts\caixas_yolo\14_criar_dataset_yolo.py
scripts\caixas_yolo\15_treinar_yolo.py
scripts\caixas_yolo\16_avaliar_yolo.py
scripts\caixas_yolo\17_conferir_erros_yolo.py
saidas\tabelas\06_modelos\yolo\metricas_yolo_teste.csv
```

Resultado no teste:

| cenario YOLO | threshold | acuracia | precisao contaminada | recall/sensibilidade contaminada | especificidade nao contaminada | F1 contaminada | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| regra atual melhor deteccao | 0.25 | 0.6038 | 0.6292 | 0.8615 | 0.1951 | 0.7273 | 8 | 33 | 9 | 56 |
| melhor F1 validacao | 0.25 | 0.5943 | 0.6100 | 0.9385 | 0.0488 | 0.7394 | 2 | 39 | 4 | 61 |
| prioridade recall validacao | 0.25 | 0.5943 | 0.6100 | 0.9385 | 0.0488 | 0.7394 | 2 | 39 | 4 | 61 |

Resumo por origem no teste, para o YOLO:

| origem | quantidade | acuracia | precisao contaminada | recall contaminada | F1 contaminada | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Micro-ondas | 33 | 0.6061 | 0.6061 | 1.0000 | 0.7547 | 0 | 13 | 0 | 20 |
| Piloto | 9 | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0 | 6 | 0 | 3 |
| TESTE_2 | 64 | 0.6250 | 0.6552 | 0.9048 | 0.7600 | 2 | 20 | 4 | 38 |

Interpretacao:

- O YOLO nao superou o baseline ResNet18.
- A localizacao da semente nao criou informacao nova suficiente.
- YOLO reduziu alguns falsos positivos em um cenario, mas perdeu muitas contaminadas.
- Como as caixas sao pseudo-rotulos e o ganho foi baixo, nao ha justificativa forte para treinar YOLO maior agora.

### 6.3 ResNet18 usando recortes da semente

Objetivo:

- usar os recortes gerados pelas caixas automaticas;
- remover fundo, regua, pinca, bancada e etiquetas o maximo possivel;
- testar se o classificador melhora quando ve somente a semente.

Arquivos principais:

```text
scripts\recortes\18_treinar_recortes_resnet18.py
scripts\recortes\19_avaliar_recortes_resnet18.py
scripts\recortes\20_comparar_resultados_modelos.py
scripts\recortes\21_conferir_erros_recortes.py
saidas\tabelas\06_modelos\recortes\metricas_recortes_resnet18_teste.csv
saidas\tabelas\06_modelos\recortes\resumo_recortes_por_origem_teste.csv
```

Configuracao de treino usada no script 18:

```text
BATCH_SIZE = 24
NUM_WORKERS = 4
PIN_MEMORY_CUDA = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USAR_CHANNELS_LAST_CUDA = True
USAR_TF32_CUDA = True
```

O treino parou por early stopping na epoca 6. Melhor modelo salvo na epoca 2, com validacao:

```text
Validacao epoca 2:
loss = 0.6436
recall = 0.8462
especificidade = 0.2683
F1 = 0.7333
```

Resultado no teste:

| cenario | threshold | acuracia | precisao contaminada | recall/sensibilidade contaminada | especificidade nao contaminada | F1 contaminada | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.50 | 0.50 | 0.6321 | 0.6585 | 0.8308 | 0.3171 | 0.7347 | 13 | 28 | 11 | 54 |
| melhor F1 validacao | 0.35 | 0.6132 | 0.6176 | 0.9692 | 0.0488 | 0.7545 | 2 | 39 | 2 | 63 |
| prioridade recall validacao | 0.35 | 0.6132 | 0.6176 | 0.9692 | 0.0488 | 0.7545 | 2 | 39 | 2 | 63 |

Resumo por origem dos recortes:

| cenario | origem | quantidade | recall/sensibilidade | especificidade | F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.35 | Micro-ondas | 33 | 0.9500 | 0.0769 | 0.7451 | 1 | 12 | 1 | 19 |
| threshold 0.35 | Piloto | 9 | 0.6667 | 0.0000 | 0.3636 | 0 | 6 | 1 | 2 |
| threshold 0.35 | TESTE_2 | 64 | 1.0000 | 0.0455 | 0.8000 | 1 | 21 | 0 | 42 |
| threshold 0.50 | Micro-ondas | 33 | 0.8000 | 0.3846 | 0.7273 | 5 | 8 | 4 | 16 |
| threshold 0.50 | Piloto | 9 | 0.3333 | 0.1667 | 0.2222 | 1 | 5 | 2 | 1 |
| threshold 0.50 | TESTE_2 | 64 | 0.8810 | 0.3182 | 0.7872 | 7 | 15 | 5 | 37 |

Interpretacao:

- Os recortes melhoraram levemente o F1 em relacao ao baseline sensivel:
  - baseline melhor F1: 0.7529
  - recortes melhor F1: 0.7545
- A melhoria e muito pequena.
- O threshold 0.35 manteve recall alto, mas ainda gerou 39 falsos positivos.
- O threshold 0.50 reduziu falsos positivos para 28, mas aumentou falsos negativos para 11.
- Para um problema em que detectar `contaminada` e prioridade, perder 11 contaminadas e muito arriscado.

## 7. Comparacao final dos modelos

Tabela consolidada no teste:

| modelo | cenario | threshold | recall/sensibilidade | especificidade | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline imagem inteira | melhor F1 validacao | 0.25 | 0.9846 | 0.0000 | 0.7529 | 41 | 1 |
| recortes ResNet18 | melhor F1 validacao | 0.35 | 0.9692 | 0.0488 | 0.7545 | 39 | 2 |
| YOLO | melhor F1 validacao | 0.25 | 0.9385 | 0.0488 | 0.7394 | 39 | 4 |
| recortes ResNet18 | threshold 0.50 | 0.50 | 0.8308 | 0.3171 | 0.7347 | 28 | 11 |
| YOLO | regra atual | 0.25 | 0.8615 | 0.1951 | 0.7273 | 33 | 9 |

Melhor modelo para recall/sensibilidade:

- baseline ResNet18 com imagem inteira, threshold 0.25
- recall 0.9846
- apenas 1 falso negativo
- mas especificidade 0.0000

Melhor equilibrio pequeno de F1:

- ResNet18 com recortes, threshold 0.35
- F1 0.7545
- recall 0.9692
- especificidade 0.0488

Modelo mais especifico entre os testados:

- ResNet18 com recortes, threshold 0.50
- especificidade 0.3171
- mas recall cai para 0.8308
- 11 contaminadas passam como nao contaminadas

## 8. Conclusao tecnica e cientifica

A etapa atual indica que:

1. O pipeline tecnico funciona.
2. O dataset binario foi criado e auditado.
3. As caixas automaticas e os recortes foram gerados.
4. O YOLO foi treinado e avaliado, mas nao superou o baseline.
5. O classificador com recortes melhorou muito pouco o F1.
6. Os falsos positivos continuam altos.
7. A reducao forte de falsos positivos exige aumentar o threshold, mas isso aumenta demais os falsos negativos.

Conclusao principal:

> Com as imagens iniciais disponiveis, nao ha evidencia forte de separabilidade visual robusta entre `contaminada` e `nao_contaminada`.

Como o usuario conferiu visualmente os recortes e observou que as imagens sao muito parecidas, a interpretacao mais provavel e:

> O problema nao esta principalmente nas caixas ou nos recortes; o gargalo esta na fraca diferenca visual entre as classes nas fotos iniciais.

Isso nao invalida o projeto. Pelo contrario, gera uma conclusao cientifica importante:

- as imagens iniciais isoladas podem nao conter sinal visual suficiente para prever contaminacao posterior com alta confianca;
- modelos mais complexos podem aprender padroes indiretos ou acidentais, como origem, iluminacao, fundo, lote ou tratamento;
- o resultado deve ser reportado como evidencia exploratoria, nao como modelo pronto para uso.

## 9. Recomendacao para continuidade

Nao e recomendado treinar imediatamente modelos maiores, como YOLO maior ou redes muito mais pesadas, antes de mudar a pergunta ou adicionar informacao.

Proximos passos mais uteis:

1. Registrar a conclusao atual no relatorio cientifico.
2. Se continuar o projeto, testar informacoes adicionais:
   - origem/tratamento;
   - tempo de incubacao;
   - fotos posteriores;
   - variaveis das planilhas;
   - descritores simples de cor/textura.
3. Avaliar modelos separados por origem, especialmente `Micro-ondas`, `Piloto` e `TESTE_2`.
4. Conferir se as classes por origem estao balanceadas e se existe vies de lote.
5. Se houver novas fotos com sinal visual mais forte, repetir o pipeline.

## 10. Arquivos gerados mais importantes

Tabelas:

```text
saidas\tabelas\06_modelos\baseline\metricas_baseline_resnet18_teste.csv
saidas\tabelas\06_modelos\yolo\metricas_yolo_teste.csv
saidas\tabelas\06_modelos\recortes\metricas_recortes_resnet18_teste.csv
saidas\tabelas\06_modelos\comparacao\comparacao_modelos_teste.csv
saidas\tabelas\06_modelos\yolo\resumo_yolo_por_origem_teste.csv
saidas\tabelas\06_modelos\recortes\resumo_recortes_por_origem_teste.csv
saidas\tabelas\06_modelos\recortes\erros_recortes_resnet18_teste.csv
```

Figuras/conferencias:

```text
saidas\conferencia_yolo\erros\falsos_positivos_yolo.png
saidas\conferencia_yolo\erros\falsos_negativos_yolo.png
saidas\conferencia_recortes\erros\falsos_positivos_recortes_threshold_0_35.png
saidas\conferencia_recortes\erros\falsos_negativos_recortes_threshold_0_35.png
saidas\conferencia_recortes\erros\falsos_positivos_recortes_threshold_0_50.png
saidas\conferencia_recortes\erros\falsos_negativos_recortes_threshold_0_50.png
```

Modelos:

```text
saidas\modelos\baseline_resnet18_melhor.pt
saidas\modelos\recortes_resnet18_melhor.pt
saidas\yolo_runs\sementes_yolo_caixas_auto\weights\best.pt
```

## 11. Ate 10 arquivos recomendados para anexar em nova conversa

Se houver limite de 10 arquivos, enviar estes:

1. `docs\relatorio_contexto_chatgpt.md`
2. `README.md`
3. `environment.yml`
4. `scripts\baseline\06_treinar_baseline.py`
5. `scripts\baseline\07_avaliar_modelo.py`
6. `scripts\caixas_yolo\16_avaliar_yolo.py`
7. `scripts\recortes\18_treinar_recortes_resnet18.py`
8. `scripts\recortes\19_avaliar_recortes_resnet18.py`
9. `saidas\tabelas\06_modelos\comparacao\comparacao_modelos_teste.csv`
10. `saidas\tabelas\06_modelos\recortes\resumo_recortes_por_origem_teste.csv`

Se puder substituir algum script por imagens de erro, as imagens mais informativas sao:

```text
saidas\conferencia_recortes\erros\falsos_positivos_recortes_threshold_0_35.png
saidas\conferencia_recortes\erros\falsos_negativos_recortes_threshold_0_35.png
```

## 12. Frase curta para iniciar uma nova conversa

Use este texto ao abrir uma nova conversa:

```text
Estou trabalhando em um projeto de classificacao de sementes de macauba em contaminada vs nao_contaminada. As fotos foram tiradas no inicio, e a contaminacao foi observada posteriormente. A classe positiva e contaminada, e a prioridade e recall/sensibilidade. Ja testamos ResNet18 com imagem inteira, YOLO com caixas automaticas e ResNet18 com recortes. Os recortes estao corretos, mas as sementes sao visualmente muito parecidas. Leia o relatorio anexado e os CSVs de metricas para continuar a analise sem assumir que o modelo enxerga contaminacao diretamente.
```

## 13. Estado atual recomendado

Estado do projeto ao final desta etapa:

- manter o baseline e o modelo com recortes como evidencia experimental;
- nao afirmar que o modelo identifica contaminacao visual diretamente;
- nao treinar modelos maiores sem nova hipotese;
- usar a conclusao atual como base para o relatorio cientifico.


