# Plano da fase 2

Data de criacao: 02/06/2026.

## Objetivo

Mudar o projeto de classificacao binaria direta para um sistema de risco/triagem.

Em vez de responder apenas:

- `contaminada`;
- `nao_contaminada`;

a fase 2 deve responder:

- `alto_risco`;
- `baixo_risco`;
- `incerto`.

Essa formulacao combina melhor com a conclusao da fase RGB: a imagem inicial sozinha nao separou as classes com seguranca suficiente.

## Pergunta cientifica

Pergunta principal:

> E possivel combinar imagem inicial, origem, tratamento e informacoes das planilhas para estimar risco de contaminacao e apoiar a triagem das sementes?

Perguntas secundarias:

- Em quais sementes o modelo consegue decidir com confianca?
- Nas sementes decididas, qual e a taxa de acerto?
- A origem ou o tratamento explicam parte dos erros dos modelos RGB?
- Atributos simples de cor, textura e forma carregam algum sinal util?

## Arquivo central

O arquivo central da fase 2 sera:

```text
saidas\tabelas\07_fase2_triagem\tabela_mestre_v2.csv
```

Ele deve ter uma linha por semente/imagem e reunir:

- caminho da imagem;
- nome do arquivo;
- origem;
- tratamento;
- rotulo real;
- split usado na fase 1, quando existir;
- probabilidade do baseline ResNet18;
- probabilidade do modelo ResNet18 com recortes;
- resultado preliminar da triagem;
- dados adicionais das planilhas, quando houver correspondencia segura.

## Primeiro script

Criar:

```text
scripts\fase2\22_criar_tabela_mestre_v2.py
```

Entradas iniciais:

```text
saidas\tabelas\03_tabela_mestre\tabela_mestre.csv
saidas\tabelas\03_tabela_mestre\tabela_mestre_treinavel.csv
saidas\tabelas\06_modelos\baseline\predicoes_baseline_resnet18_teste.csv
saidas\tabelas\06_modelos\recortes\predicoes_recortes_resnet18_teste.csv
dados_originais\planilhas\
```

Saida:

```text
saidas\tabelas\07_fase2_triagem\tabela_mestre_v2.csv
```

## Primeira regra de triagem

Usar as probabilidades ja existentes antes de treinar qualquer modelo novo.

Regra inicial sugerida para testar:

| Probabilidade de contaminacao | Triagem |
|---:|---|
| >= 0.70 | `alto_risco` |
| <= 0.30 | `baixo_risco` |
| > 0.30 e < 0.70 | `incerto` |

Esses limites sao ponto de partida. Depois devem ser ajustados usando os CSVs de validacao e teste.

Metricas da triagem:

- cobertura: porcentagem de sementes fora da zona `incerto`;
- acuracia entre as sementes decididas;
- falsos negativos entre as sementes decididas;
- proporcao de `incerto` por origem e tratamento.

## Depois da tabela mestre v2

Ordem recomendada:

1. Gerar `tabela_mestre_v2.csv`.
2. Avaliar a regra simples de triagem com probabilidades existentes.
3. Extrair atributos simples da imagem:
   - cor media;
   - brilho;
   - contraste;
   - textura;
   - area;
   - circularidade;
   - proporcao;
   - variacao de cor.
4. Treinar modelos simples:
   - regressao logistica;
   - Random Forest;
   - SVM;
   - XGBoost, se necessario.
5. Comparar:
   - imagem inteira;
   - recortes;
   - atributos simples;
   - dados de planilha;
   - modelo hibrido.

## O que evitar agora

Evitar neste momento:

- YOLO maior;
- ResNet maior;
- ViT ou modelos pesados;
- aplicativo;
- ajuste fino extenso de hiperparametros;
- prometer classificacao automatica confiavel.

O gargalo atual parece ser informacao insuficiente na imagem RGB inicial, nao falta de potencia computacional.


