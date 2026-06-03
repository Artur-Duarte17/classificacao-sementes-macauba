from pathlib import Path
import math
import textwrap
import warnings

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# SCRIPT 13 - CONFERIR CAIXAS AUTOMATICAS
# ------------------------------------------------------------
# Objetivo:
# - Gerar grades de imagens com as caixas automaticas desenhadas
# - Facilitar a revisao visual antes de treinar YOLO
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_CAIXAS_TABELAS = PASTA_TABELAS / "05_caixas_yolo"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "conferencia_caixas" / "grades"

CLASSES = ["contaminada", "nao_contaminada"]
QUANTIDADE_POR_CLASSE = 40
SEMENTE_ALEATORIA = 42
COLUNAS_GRADE = 5
TAMANHO_THUMBNAIL = (220, 220)

LARGURA_CELULA = 300
ALTURA_CELULA = 340
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


def quebrar_legenda(texto: str) -> list[str]:
    linhas = textwrap.wrap(str(texto), width=36, break_long_words=True)

    if len(linhas) > 3:
        linhas = linhas[:3]
        linhas[-1] = linhas[-1][:33] + "..."

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


def gerar_grade(classe: str, amostras: pd.DataFrame, caminho_saida: Path):
    fonte_titulo = carregar_fonte(24)
    fonte_legenda = carregar_fonte(13)

    imagens = []

    for _, linha in amostras.iterrows():
        caminho = PASTA_PROJETO / str(linha["arquivo_anotado"])

        if not caminho.exists():
            continue

        try:
            origem_detector = ""

            if "origem_detector" in linha.index and pd.notna(linha["origem_detector"]):
                origem_detector = str(linha["origem_detector"])

            imagens.append((
                criar_thumbnail(caminho),
                str(linha["caminho_relativo_original"]),
                str(linha["status_caixa"]),
                str(linha["metodo"]),
                origem_detector,
            ))
        except Exception:
            continue

    total = len(imagens)
    colunas = min(COLUNAS_GRADE, max(total, 1))
    linhas = max(math.ceil(total / colunas), 1)

    largura_grade = MARGEM * 2 + colunas * LARGURA_CELULA + (colunas - 1) * ESPACO
    altura_grade = MARGEM * 2 + ALTURA_TITULO + linhas * ALTURA_CELULA + (linhas - 1) * ESPACO

    grade = Image.new("RGB", (largura_grade, altura_grade), COR_FUNDO)
    draw = ImageDraw.Draw(grade)
    draw.text(
        (MARGEM, MARGEM),
        f"Conferencia de caixas - {classe} ({total} imagens)",
        font=fonte_titulo,
        fill=COR_TEXTO,
    )

    for indice, (thumbnail, legenda, status, metodo, origem_detector) in enumerate(imagens):
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

        textos = [f"status: {status}"]

        if origem_detector:
            textos.append(f"detector: {origem_detector}")

        textos.extend([f"metodo: {metodo}", *quebrar_legenda(legenda)])

        for texto in textos:
            desenhar_texto_centralizado(draw, texto, centro_x, y_legenda, fonte_legenda)
            y_legenda += 17

    grade.save(caminho_saida)


def main():
    print("=" * 60)
    print("CONFERINDO CAIXAS AUTOMATICAS")
    print("=" * 60)

    caminho_caixas = PASTA_CAIXAS_TABELAS / "caixas_automaticas.csv"

    if not caminho_caixas.exists():
        print("ERRO: caixas_automaticas.csv nao encontrado.")
        print(caminho_caixas)
        print("Execute primeiro:")
        print("python scripts\\08_gerar_caixas_microondas.py")
        print("python scripts\\09_gerar_caixas_piloto_teste2.py")
        print("python scripts\\10_juntar_caixas_automaticas.py")
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_caixas)
    df = df[df["status_caixa"].isin(["ok", "fallback"])].copy()

    caminhos = []

    for classe in CLASSES:
        df_classe = df[df["classe"] == classe].copy()
        quantidade = min(QUANTIDADE_POR_CLASSE, len(df_classe))
        amostras = df_classe.sample(n=quantidade, random_state=SEMENTE_ALEATORIA) if quantidade else df_classe

        caminho_grade = PASTA_SAIDA / f"caixas_{classe}.png"
        gerar_grade(classe, amostras, caminho_grade)
        caminhos.append(caminho_grade)

    print("Arquivos gerados:")
    for caminho in caminhos:
        print(f"- {caminho}")

    print()
    print("Confira as grades antes de criar/treinar o dataset YOLO.")

if __name__ == "__main__":
    main()



