# Adequacao do escopo da proposta

Data da revisao: 2026-06-08.

## 1. Objetivo original

O projeto avalia se imagens iniciais de sementes de macauba conseguem predizer
contaminacao registrada posteriormente nas planilhas. A classe positiva e
`contaminada`. A pergunta operacional e se o sinal disponivel permite separar
amostras de alto risco, baixo risco ou incerteza sem apresentar a classificacao
como deteccao visual direta de infeccao.

## 2. Evidencias usadas

Este documento foi escrito a partir dos artefatos ja consolidados, sem criar
gerador automatico e sem recalcular modelos, thresholds ou splits.

| Artefato | Papel como evidencia |
|---|---|
| `saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv` | split original, classes e total de amostras |
| `saidas/tabelas/07_classificacao_final/comparacao_final_classificacao.csv` | resultados finais no split original |
| `saidas/tabelas/07_classificacao_final/validacao_tratamento/resumo_generalizacao_por_tratamento.csv` | agregacao micro/macro da validacao externa por tratamento |
| `saidas/tabelas/07_classificacao_final/validacao_tratamento/diagnostico_folds_validacao_por_tratamento.csv` | 12 grupos externos e tamanhos dos folds |
| `docs/relatorio_classificacao_cientifica.md` | interpretacao cientifica final da classificacao |

## 3. Base experimental

A base consolidada tem 703 amostras: 429 contaminadas e 274 nao contaminadas.
O split original esta preservado para comparabilidade com os modelos ja
treinados.

| split | contaminada | nao_contaminada | total |
|---|---:|---:|---:|
| treino | 299 | 192 | 491 |
| validacao | 65 | 41 | 106 |
| teste | 65 | 41 | 106 |

A validacao externa usa 12 grupos `experimento_tratamento`. O menor grupo
externo e `teste_2__t4`, com 3 amostras, e o maior e
`micro_ondas__controle`, com 115 amostras. Essa assimetria torna as metricas
macro instaveis e reforca a necessidade de interpretar resultados por grupo.

## 4. Matriz de aderencia ao escopo

| Item da proposta | Evidencia atual | Status |
|---|---|---|
| Organizacao da base de imagens e planilhas | scripts 00-05 e split consolidado com 703 amostras | atendido |
| Classificacao binaria contaminada vs nao_contaminada | modelos avaliados no split original e validacao externa | atendido |
| Uso de imagens iniciais | ResNet18, YOLO/caixas, recortes, MobileNetV2 e atributos visuais | atendido |
| Avaliacao com metricas de classificacao | recall, especificidade, F1, balanced accuracy e MCC | atendido |
| Comparacao com metadados | baseline de origem/tratamento/pasta como diagnostico de vies | atendido |
| Controle de lote/tratamento | leave-one-experimento-tratamento-out com 12 grupos | atendido |
| Interpretacao operacional | conclusao direciona classificacao direta para triagem preventiva | atendido |
| Aplicativo final de classificacao automatica | nao recomendado pelos resultados externos | nao indicado |
| Liberacao automatica de baixo risco | ainda sem evidencia suficiente | pendente |

Status permitidos nesta matriz: `atendido`, `parcial`, `pendente`,
`nao indicado`.

## 5. Resultado do split original

No split original, a MobileNetV2 com recortes foi o melhor modelo visual em
`threshold=0,50`, com balanced accuracy 0,640, MCC 0,274, recall 0,646 e
especificidade 0,634. O baseline de metadados obteve balanced accuracy 0,664 e
MCC 0,409, mas esse resultado e interpretado como diagnostico de vies de
lote/tratamento, nao como candidato para o aplicativo.

O controle sempre-contaminada manteve recall 1,000 e especificidade 0,000. Ele
serve como referencia minima para mostrar que F1 isolado nao e criterio
suficiente.

## 6. Resultado da validacao externa

Na validacao leave-one-experimento-tratamento-out, a MobileNetV2 com recortes
caiu para balanced accuracy 0,446 e MCC -0,106 no `threshold=0,50`. O Random
Forest com threshold validado chegou a balanced accuracy 0,523 e MCC 0,061. A
queda entre split original e grupos externos indica que os modelos visuais nao
generalizaram o bastante para classificacao automatica direta em tratamentos
desconhecidos.

## 7. Diagnostico de vies de lote/tratamento

O baseline de metadados apresentou desempenho competitivo no split original e
forte associacao com origem, tratamento e pasta. Isso e evidencia de que parte
da predicao pode estar ligada ao contexto experimental, nao necessariamente a
um sinal visual biologico robusto. Por isso, metadados devem permanecer como
diagnostico de vies e controle cientifico, nao como modelo candidato ao uso
operacional.

## 8. Adequacao cientifica

O escopo permanece adequado se a conclusao for formulada como avaliacao de
viabilidade e risco, nao como entrega de um classificador automatico confiavel.
Os resultados sustentam a afirmacao de que a classificacao direta foi testada
de forma reprodutivel e que a validacao externa revelou limitacoes importantes
de generalizacao.

## 9. Ajuste de escopo recomendado

A etapa seguinte deve ser triagem preventiva, com tres saidas:

- `alto_risco`: separar ou revisar com prioridade;
- `incerto`: manter revisao manual;
- `baixo_risco`: nao usar como liberacao automatica sem nova evidencia.

Esse ajuste preserva o objetivo de apoiar a tomada de decisao, mas evita
afirmar que o modelo substitui a avaliacao experimental.

## 10. Limitacoes

Ha grupos externos muito pequenos, como `teste_2__t4` com 3 amostras. Alguns
tratamentos tambem apresentam distribuicoes de classe muito desbalanceadas. Por
isso, metricas macro podem variar bastante e devem ser lidas junto com
agregacoes micro, matrizes de confusao e diagnosticos por grupo.

## 11. Conclusao

A classificacao direta foi concluida dentro do escopo cientifico da proposta,
mas os resultados externos nao sustentam uso automatico em tratamentos
desconhecidos. O projeto deve ser apresentado como uma analise reprodutivel que
identificou sinal limitado, risco de vies de lote/tratamento e justificativa
metodologica para avancar para triagem preventiva conservadora.

Este documento nao afirma aprovacao do orientador nem substitui avaliacao
institucional da proposta.
