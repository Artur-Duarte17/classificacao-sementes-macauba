# classificacao-sementes-macauba

Projeto de iniciacao cientifica/prototipo rapido para classificar imagens de sementes de macauba em duas classes:

- `contaminada`
- `nao_contaminada`

A classe positiva do problema e `contaminada`. A metrica mais importante para a proxima fase e o recall/sensibilidade dessa classe.

## Contexto cientifico

As imagens foram tiradas no comeco dos tratamentos. A contaminacao foi observada depois e registrada nas planilhas.

Portanto, o modelo nao deve ser descrito como uma IA que enxerga infeccao diretamente na imagem inicial. O enquadramento correto e: modelo preditivo que procura padroes visuais associados a contaminacao registrada posteriormente.

## Estrutura local esperada

```text
C:\Projetos\sementes_ia
  dados_originais\
    imagens\
    planilhas\
  scripts\
  saidas\
    dataset_binario\
      contaminada\
      nao_contaminada\
```

`dados_originais/` e `saidas/` nao devem ser enviados para o GitHub, porque contem imagens, tabelas derivadas e arquivos pesados/locais.

## Ambiente

Ative o ambiente conda:

```powershell
conda activate sementes_ia
```

Dependencias basicas do projeto:

```powershell
conda env update -f environment.yml
```

Para treino com GPU, instale PyTorch com CUDA seguindo o seletor oficial:

https://pytorch.org/get-started/locally/

No Windows com NVIDIA, selecione:

- Stable
- Windows
- Pip
- Python
- CUDA recomendada pelo site

Depois confira se a GPU foi detectada:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Ordem dos scripts

A ordem abaixo separa o projeto em blocos. Rode apenas o bloco necessario para a etapa atual.

### 1. Preparar dados

```powershell
python scripts\00_inventario_imagens.py
python scripts\01_ler_planilhas_rotulos.py
python scripts\02_criar_rotulos_planilhas.py
python scripts\03_criar_tabela_mestre.py
python scripts\04_criar_dataset_binario.py
python scripts\05_conferir_amostras_dataset.py
```

Esses scripts criam o dataset binario e a conferencia visual inicial. Eles nao treinam modelo.

### 2. Baseline com imagem inteira

```powershell
python scripts\06_treinar_baseline.py
python scripts\07_avaliar_modelo.py
```

Este e o resultado de comparacao principal. Ele usa a imagem inteira, sem caixas.

Saidas principais:

- `saidas\modelos\baseline_resnet18_melhor.pt`
- `saidas\tabelas\metricas_baseline_resnet18_teste.csv`
- `saidas\tabelas\predicoes_baseline_resnet18_teste.csv`
- `saidas\figuras\matriz_confusao_baseline_resnet18_teste.png`

### 3. Caixas, recortes e YOLO

Esta etapa usa caixas geradas automaticamente por OpenCV. Elas sao pseudo-rotulos, entao precisam ser conferidas visualmente antes do treino YOLO.

```powershell
python scripts\08_gerar_caixas_microondas.py
python scripts\09_gerar_caixas_piloto_teste2.py
python scripts\10_juntar_caixas_automaticas.py
python scripts\13_conferir_caixas_automaticas.py
```

O script `08_gerar_caixas_microondas.py` trata somente imagens `Micro-ondas__`.
O script `09_gerar_caixas_piloto_teste2.py` trata imagens `Piloto__` e `TESTE_2__`, que usam outro padrao visual.
O script `10_juntar_caixas_automaticas.py` junta os relatorios em `saidas\tabelas\caixas_automaticas.csv`.

Depois confira as grades em:

```text
saidas\conferencia_caixas\grades
```

Se algumas caixas precisarem de ajuste manual, marque e aplique os ajustes:

```powershell
python scripts\11_marcar_ajustes_manuais_caixas.py --filtro TESTE_2__T6__
python scripts\12_aplicar_ajustes_manuais_caixas.py
python scripts\13_conferir_caixas_automaticas.py
```

Os ajustes manuais ficam em `saidas\tabelas\caixas_ajustes_manuais.csv`.

Se as caixas estiverem boas, crie o dataset YOLO, treine e avalie:

```powershell
python scripts\14_criar_dataset_yolo.py
python scripts\15_treinar_yolo.py
python scripts\16_avaliar_yolo.py
python scripts\17_conferir_erros_yolo.py
```

Saidas principais:

- `saidas\tabelas\caixas_automaticas.csv`
- `saidas\tabelas\caixas_microondas.csv`
- `saidas\tabelas\caixas_piloto_teste2.csv`
- `saidas\dataset_recortado\`
- `saidas\conferencia_caixas\`
- `saidas\yolo_dataset\`
- `saidas\yolo_runs\`
- `saidas\conferencia_yolo\erros\`

### 4. Classificador usando recortes

Esta e a proxima etapa recomendada antes de treinar um YOLO maior. Ela usa os recortes ja criados em `saidas\dataset_recortado\`, mas treina um classificador ResNet18 direto nesses recortes.

```powershell
python scripts\18_treinar_recortes_resnet18.py
python scripts\19_avaliar_recortes_resnet18.py
python scripts\20_comparar_resultados_modelos.py
python scripts\21_conferir_erros_recortes.py
```

Objetivo deste bloco:

- testar se remover fundo, regua e bancada melhora o classificador;
- comparar contra o baseline de imagem inteira;
- medir recall/sensibilidade da classe `contaminada`;
- medir especificidade da classe `nao_contaminada`, para entender os falsos positivos.

Saidas principais:

- `saidas\modelos\recortes_resnet18_melhor.pt`
- `saidas\tabelas\metricas_recortes_resnet18_teste.csv`
- `saidas\tabelas\predicoes_recortes_resnet18_teste.csv`
- `saidas\tabelas\curva_threshold_recortes_resnet18_validacao.csv`
- `saidas\tabelas\comparacao_modelos_teste.csv`
- `saidas\tabelas\resumo_recortes_por_origem_teste.csv`
- `saidas\conferencia_recortes\erros\`
- `saidas\figuras\matriz_confusao_recortes_resnet18_teste.png`
- `saidas\figuras\curva_threshold_recortes_resnet18_validacao.png`

## GitHub

Este repositorio deve ser privado.

Arquivos que devem ir para o GitHub:

- `scripts/`
- `README.md`
- `.gitignore`
- `environment.yml`

Arquivos que nao devem ir:

- `dados_originais/`
- `saidas/`
- imagens
- modelos `.pt`
- arquivos `.zip`
- documentos pessoais ou relatorios em `.docx`/`.pdf`

Criacao sugerida se voce tiver o GitHub CLI (`gh`) instalado:

```powershell
git init
git add .gitignore README.md environment.yml scripts
git commit -m "Estrutura inicial do projeto"
gh repo create classificacao-sementes-macauba --private --source . --remote origin --push
```

Se voce nao tiver o `gh`, crie primeiro um repositorio privado vazio no site do GitHub com o nome `classificacao-sementes-macauba`. Depois rode:

```powershell
git init
git add .gitignore README.md environment.yml scripts
git commit -m "Estrutura inicial do projeto"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/classificacao-sementes-macauba.git
git push -u origin main
```
