from pathlib import Path
import math
import textwrap
import warnings

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# SCRIPT 13 - CONFERIR ERROS DO YOLO
# ------------------------------------------------------------
# Objetivo:
# - Gerar grades visuais dos falsos positivos e falsos negativos
# - Facilitar a interpretacao dos erros do YOLO no teste
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "conferencia_yolo" / "erros"

CAMINHO_PREDICOES = PASTA_TABELAS / "predicoes_yolo_teste.csv"

SEMENTE_ALEATORIA = 42
QUANTIDADE_MAXIMA = 40
COLUNAS_GRADE = 5
TAMANHO_THUMBNAIL = (220, 220)

LARGURA_CELULA = 330
ALTURA_CELULA = 355
MARGEM = 24
ESPACO = 14
ALTURA_TITULO = 54
COR_FUNDO = (255, 255, 255)
COR_TEXTO = (30, 30, 30)
COR_BORDA = (210, 210, 210)

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def carregar_fonte(tamanho: int):
    for nome_fonte in ["arial.ttf", "DejaVuSans.ttf", "Calibri.ttf"]:
        try:
            return ImageFont.truetype(nome_fonte, tamanho)
        except OSError:
            pass

    return ImageFont.load_default()


def quebrar_texto(texto: str, largura: int = 40) -> list[str]:
    linhas = textwrap.wrap(str(texto), width=largura, break_long_words=True)

    if len(linhas) > 2:
        linhas = linhas[:2]
        linhas[-1] = linhas[-1][:largura - 3] + "..."

    return linhas


def criar_thumbnail(caminho_imagem: Path) -> Image.Image:
    with Image.open(caminho_imagem) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail(TAMANHO_THUMBNAIL, Image.Resampling.LANCZOS)

        thumbnail = Image.new("RGB", TAMANHO_THUMBNAIL, COR_FUNDO)
        posicao_x = (TAMANHO_THUMBNAIL[0] - img.width) // 2
        posicao_y = (TAMANHO_THUMBNAIL[1] - img.height) // 2
        thumbnail.paste(img, (posicao_x, posicao_y))

    return thumbnail


def desenhar_texto_centralizado(draw, texto, centro_x, y, fonte):
    caixa = draw.textbbox((0, 0), texto, font=fonte)
    largura = caixa[2] - caixa[0]
    draw.text((centro_x - largura / 2, y), texto, font=fonte, fill=COR_TEXTO)


def montar_legenda(linha) -> list[str]:
    nome = Path(str(linha["imagem_yolo"])).name
    return [
        f"real: {linha['classe_real']}",
        f"pred: {linha['classe_predita']}",
        f"origem: {linha['origem']}",
        f"conf cont: {float(linha['conf_contaminada']):.3f}",
        f"conf nao: {float(linha['conf_nao_contaminada']):.3f}",
        *quebrar_texto(nome),
    ]


def gerar_grade(titulo: str, df: pd.DataFrame, caminho_saida: Path):
    fonte_titulo = carregar_fonte(24)
    fonte_legenda = carregar_fonte(13)

    imagens = []

    for _, linha in df.iterrows():
        caminho = Path(str(linha["imagem_yolo"]))

        if not caminho.exists():
            continue

        try:
            imagens.append((criar_thumbnail(caminho), montar_legenda(linha)))
        except Exception:
            continue

    total = len(imagens)
    colunas = min(COLUNAS_GRADE, max(total, 1))
    linhas = max(math.ceil(total / colunas), 1)

    largura_grade = MARGEM * 2 + colunas * LARGURA_CELULA + (colunas - 1) * ESPACO
    altura_grade = MARGEM * 2 + ALTURA_TITULO + linhas * ALTURA_CELULA + (linhas - 1) * ESPACO

    grade = Image.new("RGB", (largura_grade, altura_grade), COR_FUNDO)
    draw = ImageDraw.Draw(grade)
    draw.text((MARGEM, MARGEM), f"{titulo} ({total} imagens)", font=fonte_titulo, fill=COR_TEXTO)

    for indice, (thumbnail, legenda) in enumerate(imagens):
        coluna = indice % colunas
        linha = indice // colunas

        x = MARGEM + coluna * (LARGURA_CELULA + ESPACO)
        y = MARGEM + ALTURA_TITULO + linha * (ALTURA_CELULA + ESPACO)

        x_img = x + (LARGURA_CELULA - TAMANHO_THUMBNAIL[0]) // 2
        y_img = y

        draw.rectangle(
            [x_img - 1, y_img - 1, x_img + TAMANHO_THUMBNAIL[0], y_img + TAMANHO_THUMBNAIL[1]],
            outline=COR_BORDA,
        )
        grade.paste(thumbnail, (x_img, y_img))

        centro_x = x + LARGURA_CELULA // 2
        y_legenda = y_img + TAMANHO_THUMBNAIL[1] + 8

        for texto in legenda:
            desenhar_texto_centralizado(draw, texto, centro_x, y_legenda, fonte_legenda)
            y_legenda += 17

    grade.save(caminho_saida)


def selecionar_amostras(df: pd.DataFrame) -> pd.DataFrame:
    quantidade = min(QUANTIDADE_MAXIMA, len(df))

    if quantidade == 0:
        return df

    return df.sample(n=quantidade, random_state=SEMENTE_ALEATORIA)


def main():
    print("=" * 60)
    print("CONFERINDO ERROS DO YOLO")
    print("=" * 60)

    if not CAMINHO_PREDICOES.exists():
        print("ERRO: predicoes_yolo_teste.csv nao encontrado.")
        print("Execute primeiro: python scripts\\12_avaliar_yolo.py")
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CAMINHO_PREDICOES)
    colunas_obrigatorias = [
        "classe_real",
        "classe_predita",
        "conf_contaminada",
        "conf_nao_contaminada",
        "origem",
        "imagem_yolo",
    ]
    colunas_faltando = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]

    if colunas_faltando:
        print("ERRO: predicoes_yolo_teste.csv esta no formato antigo.")
        print("Execute primeiro: python scripts\\12_avaliar_yolo.py")
        print("Colunas faltando:")
        for coluna in colunas_faltando:
            print(f"- {coluna}")
        return

    falsos_positivos = df[
        (df["classe_real"] == "nao_contaminada")
        & (df["classe_predita"] == "contaminada")
    ].copy()
    falsos_negativos = df[
        (df["classe_real"] == "contaminada")
        & (df["classe_predita"] == "nao_contaminada")
    ].copy()

    caminho_fp = PASTA_SAIDA / "falsos_positivos_yolo.png"
    caminho_fn = PASTA_SAIDA / "falsos_negativos_yolo.png"

    gerar_grade("Falsos positivos YOLO", selecionar_amostras(falsos_positivos), caminho_fp)
    gerar_grade("Falsos negativos YOLO", selecionar_amostras(falsos_negativos), caminho_fn)

    print("Resumo:")
    print(f"- Falsos positivos: {len(falsos_positivos)}")
    print(f"- Falsos negativos: {len(falsos_negativos)}")
    print()
    print("Arquivos gerados:")
    print(f"- {caminho_fp}")
    print(f"- {caminho_fn}")


if __name__ == "__main__":
    main()
