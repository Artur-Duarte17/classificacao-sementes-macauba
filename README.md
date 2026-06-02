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

Scripts ja existentes:

```powershell
python scripts\00_inventario_imagens.py
python scripts\01_ler_planilhas_rotulos.py
python scripts\02_criar_rotulos_planilhas.py
python scripts\03_criar_tabela_mestre.py
python scripts\04_criar_dataset_binario.py
python scripts\05_conferir_amostras_dataset.py
```

Proxima fase, treino e avaliacao:

```powershell
python scripts\06_treinar_baseline.py
python scripts\07_avaliar_modelo.py
```

O script de treino salva o melhor modelo em `saidas/modelos/`.

O script de avaliacao salva metricas, predicoes e figuras em `saidas/tabelas/` e `saidas/figuras/`.

## YOLO com caixas automaticas

Esta etapa usa caixas geradas automaticamente por OpenCV. Elas sao pseudo-rotulos e precisam ser conferidas visualmente antes do treino.

Instale/atualize as dependencias:

```powershell
conda activate sementes_ia
conda env update -f environment.yml
```

Sequencia recomendada:

```powershell
python scripts\08_gerar_caixas_microondas.py
python scripts\08b_gerar_caixas_piloto_teste2.py
python scripts\08c_juntar_caixas_automaticas.py
python scripts\09_conferir_caixas_automaticas.py
```

O script `08_gerar_caixas_microondas.py` trata somente imagens `Micro-ondas__`.
O script `08b` trata imagens `Piloto__` e `TESTE_2__`, que usam outro padrao visual.
O script `08c` junta os dois relatorios em `saidas\tabelas\caixas_automaticas.csv`.

Depois confira as grades em:

```text
saidas\conferencia_caixas\grades
```

Se algumas caixas precisarem de ajuste manual, marque e aplique os ajustes:

```powershell
python scripts\08d_marcar_ajustes_manuais_caixas.py --filtro TESTE_2__T6__
python scripts\08e_aplicar_ajustes_manuais_caixas.py
python scripts\09_conferir_caixas_automaticas.py
```

Os ajustes manuais ficam em `saidas\tabelas\caixas_ajustes_manuais.csv`.

Se as caixas estiverem boas, crie o dataset YOLO e treine:

```powershell
python scripts\10_criar_dataset_yolo.py
python scripts\11_treinar_yolo.py
python scripts\12_avaliar_yolo.py
```

Saidas principais:

- `saidas\tabelas\caixas_automaticas.csv`
- `saidas\tabelas\caixas_microondas.csv`
- `saidas\tabelas\caixas_piloto_teste2.csv`
- `saidas\dataset_recortado\`
- `saidas\conferencia_caixas\`
- `saidas\yolo_dataset\`
- `saidas\yolo_runs\`

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
