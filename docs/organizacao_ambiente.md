# Organizacao do ambiente

Data da revisao: 02/06/2026.

Este documento separa o que e necessario para continuar a fase 2 do projeto do que e historico, pesado ou apenas apoio da fase 1.

## Direcao recomendada

Seguir o plano de mudar de classificacao binaria direta para modelo de risco/triagem:

- alto risco de contaminacao;
- baixo risco de contaminacao;
- incerto / revisar manualmente.

Antes de treinar modelos novos, o proximo arquivo central deve ser:

```text
saidas\tabelas\07_fase2_triagem\tabela_mestre_v2.csv
```

O script inicial recomendado para a nova fase e:

```text
scripts\fase2\22_criar_tabela_mestre_v2.py
```

## O que esta ativo agora

Esses arquivos e pastas devem ficar no centro do trabalho:

```text
README.md
environment.yml
scripts\
docs\
dados_originais\planilhas\
saidas\tabelas\03_tabela_mestre\tabela_mestre.csv
saidas\tabelas\03_tabela_mestre\tabela_mestre_treinavel.csv
saidas\tabelas\06_modelos\recortes\predicoes_recortes_resnet18_teste.csv
saidas\tabelas\06_modelos\baseline\predicoes_baseline_resnet18_teste.csv
saidas\tabelas\06_modelos\comparacao\comparacao_modelos_teste.csv
saidas\tabelas\06_modelos\recortes\metricas_recortes_resnet18_teste.csv
saidas\tabelas\06_modelos\recortes\resumo_recortes_por_origem_teste.csv
```

Motivo: esses arquivos permitem construir a tabela mestre v2 juntando imagem, origem, tratamento, rotulo, predicoes e informacoes das planilhas.

## O que e dado-fonte e nao deve ser apagado

```text
dados_originais\
```

Conteudo observado:

- `dados_originais\imagens\`: imagens originais dos experimentos, cerca de 3,9 GB;
- `dados_originais\planilhas\`: 3 planilhas originais, pequenas, muito importantes para a fase 2.

Recomendacao: manter no lugar. Sao a fonte primaria do projeto.

## O que e resultado pesado da fase 1

```text
saidas\
```

Resumo observado:

| Pasta | Uso | Tamanho aproximado |
|---|---|---:|
| `saidas\yolo_dataset\` | dataset gerado para YOLO | 4,85 GB |
| `saidas\conferencia_caixas\` | imagens de conferencia das caixas | 4,63 GB |
| `saidas\dataset_binario\` | dataset binario copiado das imagens originais | 3,75 GB |
| `saidas\dataset_recortado\` | recortes das sementes | 1,05 GB |
| `saidas\modelos\` | modelos ResNet treinados | 85 MB |
| `saidas\yolo_runs\` | saida do treino YOLO | 14 MB |
| `saidas\tabelas\` | CSVs de inventario, rotulos, metricas e predicoes | 1,5 MB |
| `saidas\figuras\` | figuras finais | < 1 MB |

Recomendacao: nao apagar agora. Para a fase 2, as tabelas sao essenciais e os datasets pesados podem ser regenerados, mas ainda sao uteis se for necessario auditar os experimentos da fase 1.

## O que pode ser arquivado depois

Pode sair da area principal, desde que a fase 1 ja esteja documentada e o usuario confirme:

```text
saidas\yolo_dataset\
saidas\conferencia_caixas\
saidas\dataset_binario\
saidas\dataset_recortado\
saidas\conferencia_yolo\
saidas\conferencia_recortes\
saidas\amostras_conferencia\
```

Essas pastas sao derivadas. Elas ocupam muito espaco e podem ser regeneradas pelos scripts, mas arquivar em vez de apagar e mais seguro.

Sugestao de destino local:

```text
C:\Projetos\sementes_ia_arquivo_fase1\
```

## O que parece duplicado ou confuso

### Repositorio dentro do repositorio

Existe a pasta:

```text
classificacao-sementes-macauba\
```

Ela contem praticamente so outro `.git` e um `.gitattributes`. Parece ser um repositorio criado ou clonado por engano dentro do projeto principal.

Recomendacao: arquivar fora da raiz do projeto ou remover depois de confirmar que nao ha nada util dentro. Ela ja esta no `.gitignore`, entao nao vai para o GitHub, mas pode confundir.

### Pesos YOLO soltos na raiz

Existem arquivos:

```text
yolo11n.pt
yolo26n.pt
```

O script `scripts\caixas_yolo\15_treinar_yolo.py` referencia `yolo11n.pt`. Portanto, se for mover esse arquivo, o script tambem precisa ser ajustado.

`yolo26n.pt` nao apareceu como referencia nos scripts. Pode ser arquivo baixado por engano ou tentativa anterior.

Recomendacao: manter `yolo11n.pt` enquanto o script 15 existir como historico reexecutavel. Arquivar `yolo26n.pt` se for confirmado que nao sera usado.

## Estrutura mental para seguir

### Scripts em pacotes

Os scripts foram reorganizados por etapa:

```text
scripts\
  preparacao\
  baseline\
  caixas_yolo\
  recortes\
  fase2\
```

Use sempre o caminho completo do pacote ao executar. Exemplo:

```powershell
python scripts\fase2\22_criar_tabela_mestre_v2.py
```

### Tabelas em pacotes

Os CSVs de `saidas\tabelas\` foram reorganizados assim:

| Pasta | Funcao |
|---|---|
| `01_inventario\` | inventario das imagens e diagnosticos de arquivos |
| `02_planilhas_rotulos\` | inspecao das planilhas, previas e rotulos consolidados |
| `03_tabela_mestre\` | tabela mestre, tabela treinavel e cruzamentos sem par |
| `04_dataset_split\` | relatorio do dataset binario e divisao treino/validacao/teste |
| `05_caixas_yolo\` | pseudo-caixas, ajustes e relatorio do dataset YOLO |
| `06_modelos\baseline\` | resultados tabulares do baseline ResNet18 |
| `06_modelos\yolo\` | resultados tabulares do YOLO |
| `06_modelos\recortes\` | resultados tabulares do classificador com recortes |
| `06_modelos\comparacao\` | comparacao consolidada entre modelos |
| `07_fase2_triagem\` | tabela mestre v2 e resumo de triagem |

Nenhum CSV foi apagado nessa reorganizacao, porque todos tinham utilidade de entrada, auditoria, reproducao ou resumo.

### Base versionada

Vai para GitHub:

```text
README.md
environment.yml
scripts\
docs\
.gitignore
```

### Dados locais

Nao vai para GitHub:

```text
dados_originais\
saidas\
*.pt
```

### Fase atual

Foco imediato:

1. Consolidar a fase RGB no relatorio.
2. Criar `tabela_mestre_v2.csv`.
3. Testar regra de triagem com `alto_risco`, `baixo_risco` e `incerto`.
4. Extrair atributos simples de cor, textura e forma.
5. Treinar modelos simples usando imagem + contexto experimental.

## Proxima acao recomendada

Criar o script:

```text
scripts\fase2\22_criar_tabela_mestre_v2.py
```

Esse script deve partir de:

```text
saidas\tabelas\03_tabela_mestre\tabela_mestre.csv
saidas\tabelas\06_modelos\recortes\predicoes_recortes_resnet18_teste.csv
saidas\tabelas\06_modelos\baseline\predicoes_baseline_resnet18_teste.csv
dados_originais\planilhas\
```

E gerar uma tabela com uma linha por semente, contendo pelo menos:

- identificacao da imagem;
- origem;
- tratamento;
- rotulo final;
- split, se existir;
- probabilidade do baseline;
- probabilidade do modelo com recortes;
- resultado da triagem preliminar;
- campos adicionais das planilhas quando houver correspondencia segura.

## Execucao da organizacao

Ordem operacional aprovada para limpar a raiz antes da fase 2:

1. Criar `C:\Projetos\sementes_ia_arquivo_fase1\`.
2. Mover para essa pasta os derivados pesados da fase 1:
   - `saidas\yolo_dataset\`;
   - `saidas\conferencia_caixas\`;
   - `saidas\dataset_binario\`;
   - `saidas\dataset_recortado\`;
   - `saidas\conferencia_yolo\`;
   - `saidas\conferencia_recortes\`;
   - `saidas\amostras_conferencia\`.
3. Manter no projeto principal:
   - `dados_originais\`;
   - `saidas\tabelas\`;
   - `saidas\figuras\`;
   - `saidas\modelos\`;
   - `saidas\yolo_runs\`, como registro leve do treino YOLO.
4. Arquivar fora da raiz a pasta `classificacao-sementes-macauba\`, que parece ser um repositorio criado por engano dentro do projeto.
5. Arquivar `yolo26n.pt`, pois nenhum script atual referencia esse peso.
6. Manter `yolo11n.pt`, porque `scripts\caixas_yolo\15_treinar_yolo.py` referencia esse arquivo.

## Estado apos a organizacao

A pasta de arquivo foi criada em:

```text
C:\Projetos\sementes_ia_arquivo_fase1\
```

Foram arquivados:

```text
C:\Projetos\sementes_ia_arquivo_fase1\saidas\yolo_dataset\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\conferencia_caixas\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\dataset_binario\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\dataset_recortado\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\conferencia_yolo\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\conferencia_recortes\
C:\Projetos\sementes_ia_arquivo_fase1\saidas\amostras_conferencia\
C:\Projetos\sementes_ia_arquivo_fase1\classificacao-sementes-macauba\
C:\Projetos\sementes_ia_arquivo_fase1\yolo26n.pt
```

Na raiz principal, `saidas\` ficou reduzida a:

```text
saidas\figuras\
saidas\modelos\
saidas\tabelas\
saidas\yolo_runs\
```

Tambem foi criado o plano:

```text
docs\plano_fase2.md
```

E foi criada a primeira tabela da fase 2:

```text
saidas\tabelas\07_fase2_triagem\tabela_mestre_v2.csv
saidas\tabelas\07_fase2_triagem\resumo_triagem_preliminar_v2.csv
```


