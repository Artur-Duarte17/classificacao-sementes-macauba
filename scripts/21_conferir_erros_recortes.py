from pathlib import Path
import math
import textwrap
import warnings

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# ============================================================
# SCRIPT 21 - CONFERIR ERROS DOS RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Gerar grades visuais dos erros do ResNet18 treinado em recortes
# - Comparar threshold sensivel e threshold mais especifico
# - Resumir desempenho por origem das imagens
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "conferencia_recortes" / "erros"

CAMINHO_PREDICOES = PASTA_TABELAS / "predicoes_recortes_resnet18_teste.csv"
CAMINHO_ERROS = PASTA_TABELAS / "erros_recortes_resnet18_teste.csv"
CAMINHO_RESUMO_ORIGEM = PASTA_TABELAS / "resumo_recortes_por_origem_teste.csv"

CENARIOS = [
    {
        "nome": "threshold_0_35",
        "coluna_predicao": "predito_threshold_melhor_f1_validacao",
        "threshold": 0.35,
    },
    {
        "nome": "threshold_0_50",
        "coluna_predicao": "predito_threshold_0_50",
        "threshold": 0.50,
    },
]

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


def nome_origem(caminho_imagem: str) -> str:
    nome = Path(str(caminho_imagem)).name

    if nome.startswith("Micro-ondas__"):
        return "Micro-ondas"

    if nome.startswith("Piloto__"):
        return "Piloto"

    if nome.startswith("TESTE_2__"):
        return "TESTE_2"

    return "outros"


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
    nome = Path(str(linha["caminho_imagem"])).name
    return [
        f"real: {linha['classe_real']}",
        f"pred: {linha['classe_predita']}",
        f"origem: {linha['origem']}",
        f"prob cont: {float(linha['prob_contaminada']):.3f}",
        f"threshold: {float(linha['threshold']):.2f}",
        *quebrar_texto(nome),
    ]


def selecionar_amostras(df: pd.DataFrame) -> pd.DataFrame:
    quantidade = min(QUANTIDADE_MAXIMA, len(df))

    if quantidade == 0:
        return df

    return df.sample(n=quantidade, random_state=SEMENTE_ALEATORIA)


def gerar_grade(titulo: str, df: pd.DataFrame, caminho_saida: Path):
    fonte_titulo = carregar_fonte(24)
    fonte_legenda = carregar_fonte(13)

    imagens = []

    for _, linha in df.iterrows():
        caminho = PASTA_PROJETO / str(linha["caminho_imagem"])

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


def calcular_metricas(df: pd.DataFrame, coluna_predicao: str) -> dict:
    y_real = (df["classe_real"] == "contaminada").astype(int)
    y_pred = (df[coluna_predicao] == "contaminada").astype(int)

    precisao, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()

    return {
        "quantidade": int(len(df)),
        "acuracia": float(accuracy_score(y_real, y_pred)),
        "precisao_contaminada": float(precisao[0]),
        "recall_contaminada": float(recall[0]),
        "sensibilidade_contaminada": float(tp / max(tp + fn, 1)),
        "especificidade_nao_contaminada": float(tn / max(tn + fp, 1)),
        "f1_contaminada": float(f1[0]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def gerar_resumo_por_origem(df: pd.DataFrame) -> pd.DataFrame:
    registros = []

    for cenario in CENARIOS:
        for origem, grupo in df.groupby("origem"):
            registros.append({
                "cenario": cenario["nome"],
                "threshold": cenario["threshold"],
                "origem": origem,
                **calcular_metricas(grupo, cenario["coluna_predicao"]),
            })

    return pd.DataFrame(registros).sort_values(["cenario", "origem"])


def preparar_erros(df: pd.DataFrame) -> pd.DataFrame:
    erros = []

    for cenario in CENARIOS:
        coluna = cenario["coluna_predicao"]
        df_cenario = df.copy()
        df_cenario["cenario"] = cenario["nome"]
        df_cenario["threshold"] = cenario["threshold"]
        df_cenario["classe_predita"] = df_cenario[coluna]

        df_cenario["tipo_erro"] = ""
        df_cenario.loc[
            (df_cenario["classe_real"] == "nao_contaminada")
            & (df_cenario["classe_predita"] == "contaminada"),
            "tipo_erro",
        ] = "falso_positivo"
        df_cenario.loc[
            (df_cenario["classe_real"] == "contaminada")
            & (df_cenario["classe_predita"] == "nao_contaminada"),
            "tipo_erro",
        ] = "falso_negativo"

        erros.append(df_cenario[df_cenario["tipo_erro"] != ""])

    return pd.concat(erros, ignore_index=True)


def main():
    print("=" * 60)
    print("CONFERINDO ERROS DOS RECORTES")
    print("=" * 60)

    if not CAMINHO_PREDICOES.exists():
        print("ERRO: predicoes_recortes_resnet18_teste.csv nao encontrado.")
        print("Execute primeiro: python scripts\\19_avaliar_recortes_resnet18.py")
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CAMINHO_PREDICOES)
    df["origem"] = df["caminho_imagem"].apply(nome_origem)

    colunas_obrigatorias = [
        "caminho_imagem",
        "classe_real",
        "prob_contaminada",
        "predito_threshold_0_50",
        "predito_threshold_melhor_f1_validacao",
    ]
    colunas_faltando = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]

    if colunas_faltando:
        print("ERRO: predicoes_recortes_resnet18_teste.csv esta incompleto.")
        print("Colunas faltando:")
        for coluna in colunas_faltando:
            print(f"- {coluna}")
        return

    df_erros = preparar_erros(df)
    df_erros.to_csv(CAMINHO_ERROS, index=False, encoding="utf-8-sig")

    df_resumo = gerar_resumo_por_origem(df)
    df_resumo.to_csv(CAMINHO_RESUMO_ORIGEM, index=False, encoding="utf-8-sig")

    print("Resumo por origem:")
    print(df_resumo.to_string(index=False))
    print()

    for cenario in CENARIOS:
        nome = cenario["nome"]
        df_cenario = df_erros[df_erros["cenario"] == nome]
        falsos_positivos = df_cenario[df_cenario["tipo_erro"] == "falso_positivo"].copy()
        falsos_negativos = df_cenario[df_cenario["tipo_erro"] == "falso_negativo"].copy()

        caminho_fp = PASTA_SAIDA / f"falsos_positivos_recortes_{nome}.png"
        caminho_fn = PASTA_SAIDA / f"falsos_negativos_recortes_{nome}.png"

        gerar_grade(
            f"Falsos positivos recortes - {nome}",
            selecionar_amostras(falsos_positivos),
            caminho_fp,
        )
        gerar_grade(
            f"Falsos negativos recortes - {nome}",
            selecionar_amostras(falsos_negativos),
            caminho_fn,
        )

        print(f"Cenario {nome}:")
        print(f"- Falsos positivos: {len(falsos_positivos)}")
        print(f"- Falsos negativos: {len(falsos_negativos)}")
        print(f"- {caminho_fp}")
        print(f"- {caminho_fn}")
        print()

    print("Arquivos de tabela gerados:")
    print(f"- {CAMINHO_ERROS}")
    print(f"- {CAMINHO_RESUMO_ORIGEM}")


if __name__ == "__main__":
    main()
