from pathlib import Path
import math
import textwrap
import warnings

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# SCRIPT 05 - CONFERIR AMOSTRAS DO DATASET BINARIO
# ------------------------------------------------------------
# Objetivo:
# - Ler o relatorio criado pelo script 04
# - Selecionar amostras aleatorias de cada classe
# - Gerar grades visuais para conferencia manual
# - Salvar um CSV com as imagens selecionadas
#
# Este script NAO treina IA.
# Ele apenas ajuda a conferir visualmente o dataset.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "amostras_conferencia"

CLASSES = ["contaminada", "nao_contaminada"]

QUANTIDADE_POR_CLASSE = 40
SEMENTE_ALEATORIA = 42
COLUNAS_GRADE = 5
TAMANHO_THUMBNAIL = (180, 180)

LARGURA_CELULA = 260
ALTURA_CELULA = 270
MARGEM = 24
ESPACO = 14
ALTURA_TITULO = 54
ALTURA_LEGENDA = 64
COR_FUNDO = (255, 255, 255)
COR_TEXTO = (30, 30, 30)
COR_BORDA = (210, 210, 210)

# As imagens sao locais e fazem parte do proprio experimento.
# O aviso aparece em fotos muito grandes, mas nao impede a conferencia.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def carregar_fonte(tamanho: int):
    """
    Tenta carregar uma fonte comum. Se nao conseguir, usa a fonte padrao.
    """
    nomes_fontes = ["arial.ttf", "DejaVuSans.ttf", "Calibri.ttf"]

    for nome_fonte in nomes_fontes:
        try:
            return ImageFont.truetype(nome_fonte, tamanho)
        except OSError:
            pass

    return ImageFont.load_default()


def tamanho_texto(draw: ImageDraw.ImageDraw, texto: str, fonte) -> tuple[int, int]:
    """
    Calcula largura e altura de um texto desenhado com Pillow.
    """
    caixa = draw.textbbox((0, 0), texto, font=fonte)
    largura = caixa[2] - caixa[0]
    altura = caixa[3] - caixa[1]
    return largura, altura


def quebrar_legenda(texto: str) -> list[str]:
    """
    Quebra uma legenda longa em poucas linhas para caber na celula.
    """
    linhas = textwrap.wrap(str(texto), width=32, break_long_words=True)

    if len(linhas) > 3:
        linhas = linhas[:3]
        linhas[-1] = linhas[-1][:29] + "..."

    return linhas


def criar_thumbnail(caminho_imagem: Path) -> Image.Image:
    """
    Abre uma imagem e cria uma miniatura centralizada em fundo branco.
    """
    with Image.open(caminho_imagem) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail(TAMANHO_THUMBNAIL, Image.Resampling.LANCZOS)

        thumbnail = Image.new("RGB", TAMANHO_THUMBNAIL, COR_FUNDO)
        posicao_x = (TAMANHO_THUMBNAIL[0] - img.width) // 2
        posicao_y = (TAMANHO_THUMBNAIL[1] - img.height) // 2
        thumbnail.paste(img, (posicao_x, posicao_y))

    return thumbnail


def desenhar_texto_centralizado(
    draw: ImageDraw.ImageDraw,
    texto: str,
    centro_x: int,
    y: int,
    fonte,
    cor=COR_TEXTO,
):
    """
    Desenha uma linha de texto centralizada em relacao ao eixo X informado.
    """
    largura, _ = tamanho_texto(draw, texto, fonte)
    draw.text((centro_x - largura / 2, y), texto, font=fonte, fill=cor)


def gerar_grade(classe: str, amostras: pd.DataFrame, caminho_saida: Path) -> list[dict]:
    """
    Gera uma grade PNG para uma classe e retorna os registros do CSV.
    """
    fonte_titulo = carregar_fonte(24)
    fonte_legenda = carregar_fonte(13)
    registros = []
    imagens_validas = []

    for _, linha in amostras.iterrows():
        arquivo_copiado = Path(str(linha["arquivo_copiado"]))
        legenda = str(linha["caminho_relativo_original"])
        status = "ok"
        erro = ""

        try:
            thumbnail = criar_thumbnail(arquivo_copiado)
            imagens_validas.append((thumbnail, legenda))
        except Exception as e:
            status = "erro"
            erro = str(e)

        registros.append({
            "classe": classe,
            "caminho_relativo_original": legenda,
            "arquivo_copiado": str(arquivo_copiado),
            "status": status,
            "erro": erro,
        })

    total_validas = len(imagens_validas)
    colunas = min(COLUNAS_GRADE, max(total_validas, 1))
    linhas = max(math.ceil(total_validas / colunas), 1)

    largura_grade = (
        MARGEM * 2
        + colunas * LARGURA_CELULA
        + (colunas - 1) * ESPACO
    )
    altura_grade = (
        MARGEM * 2
        + ALTURA_TITULO
        + linhas * ALTURA_CELULA
        + (linhas - 1) * ESPACO
    )

    grade = Image.new("RGB", (largura_grade, altura_grade), COR_FUNDO)
    draw = ImageDraw.Draw(grade)

    titulo = f"Amostras da classe {classe} ({total_validas} imagens validas)"
    draw.text((MARGEM, MARGEM), titulo, font=fonte_titulo, fill=COR_TEXTO)

    if total_validas == 0:
        mensagem = "Nenhuma imagem valida para montar a grade."
        draw.text(
            (MARGEM, MARGEM + ALTURA_TITULO),
            mensagem,
            font=fonte_legenda,
            fill=COR_TEXTO,
        )
        grade.save(caminho_saida)
        return registros

    for indice, (thumbnail, legenda) in enumerate(imagens_validas):
        coluna = indice % colunas
        linha = indice // colunas

        x = MARGEM + coluna * (LARGURA_CELULA + ESPACO)
        y = MARGEM + ALTURA_TITULO + linha * (ALTURA_CELULA + ESPACO)

        x_imagem = x + (LARGURA_CELULA - TAMANHO_THUMBNAIL[0]) // 2
        y_imagem = y

        draw.rectangle(
            [
                x_imagem - 1,
                y_imagem - 1,
                x_imagem + TAMANHO_THUMBNAIL[0],
                y_imagem + TAMANHO_THUMBNAIL[1],
            ],
            outline=COR_BORDA,
        )
        grade.paste(thumbnail, (x_imagem, y_imagem))

        centro_x = x + LARGURA_CELULA // 2
        y_legenda = y_imagem + TAMANHO_THUMBNAIL[1] + 10

        for linha_legenda in quebrar_legenda(legenda):
            desenhar_texto_centralizado(
                draw,
                linha_legenda,
                centro_x,
                y_legenda,
                fonte_legenda,
            )
            y_legenda += 17

    grade.save(caminho_saida)
    return registros


def selecionar_amostras(df: pd.DataFrame, classe: str) -> pd.DataFrame:
    """
    Seleciona ate QUANTIDADE_POR_CLASSE imagens de uma classe.
    """
    df_classe = df[df["classe"] == classe].copy()
    quantidade = min(QUANTIDADE_POR_CLASSE, len(df_classe))

    if quantidade == 0:
        return df_classe

    return df_classe.sample(
        n=quantidade,
        random_state=SEMENTE_ALEATORIA,
    ).reset_index(drop=True)


def main():
    print("=" * 60)
    print("CONFERENCIA DE AMOSTRAS DO DATASET BINARIO")
    print("=" * 60)

    caminho_relatorio = PASTA_DATASET_TABELAS / "relatorio_copia_dataset_binario.csv"

    if not caminho_relatorio.exists():
        print("ERRO: relatorio_copia_dataset_binario.csv nao encontrado.")
        print(caminho_relatorio)
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_relatorio)

    colunas_obrigatorias = {
        "classe",
        "caminho_relativo_original",
        "arquivo_copiado",
        "status_copia",
    }
    colunas_faltando = colunas_obrigatorias - set(df.columns)

    if colunas_faltando:
        print("ERRO: o relatorio nao contem as colunas esperadas:")
        print(", ".join(sorted(colunas_faltando)))
        return

    df = df[
        (df["status_copia"] == "ok")
        & (df["classe"].isin(CLASSES))
    ].copy()

    print(f"Registros validos no relatorio de copia: {len(df)}")
    print(f"Amostras por classe: ate {QUANTIDADE_POR_CLASSE}")
    print()

    registros_finais = []
    caminhos_grades = []

    for classe in CLASSES:
        amostras = selecionar_amostras(df, classe)
        caminho_grade = PASTA_SAIDA / f"amostras_{classe}.png"

        print(f"Gerando grade da classe {classe}: {len(amostras)} amostras")

        registros_classe = gerar_grade(classe, amostras, caminho_grade)
        registros_finais.extend(registros_classe)
        caminhos_grades.append(caminho_grade)

    relatorio_amostras = pd.DataFrame(registros_finais)
    caminho_csv = PASTA_SAIDA / "amostras_selecionadas.csv"
    relatorio_amostras.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    if len(relatorio_amostras) > 0:
        resumo = (
            relatorio_amostras.groupby(["classe", "status"])
            .size()
            .reset_index(name="quantidade")
            .sort_values(["classe", "status"])
        )
    else:
        resumo = pd.DataFrame(columns=["classe", "status", "quantidade"])

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(resumo.to_string(index=False))

    print()
    print("Arquivos gerados:")
    for caminho_grade in caminhos_grades:
        print(f"- {caminho_grade}")
    print(f"- {caminho_csv}")

    print()
    print("Conferencia de amostras concluida.")


if __name__ == "__main__":
    main()



