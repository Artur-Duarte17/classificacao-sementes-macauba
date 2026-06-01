from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# SCRIPT 08 - GERAR CAIXAS AUTOMATICAS
# ------------------------------------------------------------
# Objetivo:
# - Ler o dataset binario ja criado
# - Detectar automaticamente a regiao provavel da semente
# - Salvar recortes, imagens com caixas e relatorio CSV
#
# As caixas geradas aqui sao PSEUDO-ROTULOS.
# Elas precisam ser conferidas visualmente antes de treinar YOLO.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_RECORTE = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_CONFERENCIA = PASTA_PROJETO / "saidas" / "conferencia_caixas" / "imagens"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_PARA_INDICE = {"nao_contaminada": 0, "contaminada": 1}

TAMANHO_PROCESSAMENTO = 900
MARGEM_CAIXA = 0.18
AREA_MINIMA_PROPORCAO = 0.0008
AREA_MAXIMA_PROPORCAO = 0.85
PADDING_FALLBACK = 0.08

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def ler_imagem(caminho: Path):
    """
    Le imagem com suporte a caminhos do Windows contendo acentos/espacos.
    """
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    return imagem


def salvar_imagem(caminho: Path, imagem) -> bool:
    """
    Salva imagem com suporte a caminhos do Windows contendo acentos/espacos.
    """
    extensao = caminho.suffix or ".jpg"
    ok, buffer = cv2.imencode(extensao, imagem)

    if not ok:
        return False

    buffer.tofile(str(caminho))
    return True


def redimensionar_para_processamento(imagem):
    altura, largura = imagem.shape[:2]
    maior_lado = max(altura, largura)

    if maior_lado <= TAMANHO_PROCESSAMENTO:
        return imagem.copy(), 1.0

    escala = TAMANHO_PROCESSAMENTO / maior_lado
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    menor = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return menor, escala


def caixa_fallback(largura: int, altura: int):
    margem_x = int(largura * PADDING_FALLBACK)
    margem_y = int(altura * PADDING_FALLBACK)
    return margem_x, margem_y, largura - margem_x, altura - margem_y


def adicionar_margem_caixa(x1, y1, x2, y2, largura, altura):
    caixa_largura = x2 - x1
    caixa_altura = y2 - y1
    margem_x = int(caixa_largura * MARGEM_CAIXA)
    margem_y = int(caixa_altura * MARGEM_CAIXA)

    x1 = max(0, x1 - margem_x)
    y1 = max(0, y1 - margem_y)
    x2 = min(largura, x2 + margem_x)
    y2 = min(altura, y2 + margem_y)

    return x1, y1, x2, y2


def detectar_caixa(imagem):
    """
    Detecta a maior regiao visualmente diferente do fundo.

    Estrategia:
    - Reduz imagem para acelerar.
    - Usa blur e threshold de Otsu no canal de luminosidade.
    - Testa mascara normal e invertida.
    - Escolhe o maior contorno com area plausivel.
    """
    altura_original, largura_original = imagem.shape[:2]
    menor, escala = redimensionar_para_processamento(imagem)
    altura_menor, largura_menor = menor.shape[:2]

    gray = cv2.cvtColor(menor, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mascara_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), np.uint8)
    candidatos = []

    for nome_mascara, mascara in [
        ("otsu", mascara_otsu),
        ("otsu_invertida", cv2.bitwise_not(mascara_otsu)),
    ]:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        area_imagem = largura_menor * altura_menor

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            proporcao_area = area / area_imagem

            if proporcao_area < AREA_MINIMA_PROPORCAO:
                continue

            if proporcao_area > AREA_MAXIMA_PROPORCAO:
                continue

            x, y, w, h = cv2.boundingRect(contorno)
            area_caixa = w * h

            candidatos.append({
                "metodo": nome_mascara,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_contorno": area,
                "area_caixa": area_caixa,
                "proporcao_area": proporcao_area,
            })

    if not candidatos:
        x1, y1, x2, y2 = caixa_fallback(largura_original, altura_original)
        return x1, y1, x2, y2, "fallback", "sem_contorno_plausivel"

    melhor = sorted(
        candidatos,
        key=lambda item: (item["area_contorno"], item["area_caixa"]),
        reverse=True,
    )[0]

    x1_m = melhor["x"]
    y1_m = melhor["y"]
    x2_m = melhor["x"] + melhor["w"]
    y2_m = melhor["y"] + melhor["h"]

    x1 = int(x1_m / escala)
    y1 = int(y1_m / escala)
    x2 = int(x2_m / escala)
    y2 = int(y2_m / escala)

    x1, y1, x2, y2 = adicionar_margem_caixa(
        x1,
        y1,
        x2,
        y2,
        largura_original,
        altura_original,
    )

    return x1, y1, x2, y2, melhor["metodo"], ""


def caminho_relativo(caminho: Path) -> str:
    return caminho.relative_to(PASTA_PROJETO).as_posix()


def main():
    print("=" * 60)
    print("GERANDO CAIXAS AUTOMATICAS")
    print("=" * 60)

    caminho_relatorio = PASTA_TABELAS / "relatorio_copia_dataset_binario.csv"

    if not caminho_relatorio.exists():
        print("ERRO: relatorio_copia_dataset_binario.csv nao encontrado.")
        print(caminho_relatorio)
        return

    PASTA_RECORTE.mkdir(parents=True, exist_ok=True)
    PASTA_CONFERENCIA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_relatorio)
    df = df[
        (df["status_copia"] == "ok")
        & (df["classe"].isin(CLASSES))
    ].copy()

    print(f"Imagens para processar: {len(df)}")
    print()

    registros = []

    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Detectando caixas"):
        classe = str(linha["classe"])
        nome_copiado = str(linha["nome_copiado"])
        caminho_imagem = Path(str(linha["arquivo_copiado"]))

        pasta_recorte_classe = PASTA_RECORTE / classe
        pasta_conf_classe = PASTA_CONFERENCIA / classe
        pasta_recorte_classe.mkdir(parents=True, exist_ok=True)
        pasta_conf_classe.mkdir(parents=True, exist_ok=True)

        caminho_recorte = pasta_recorte_classe / nome_copiado
        caminho_anotada = pasta_conf_classe / nome_copiado

        status = "ok"
        erro = ""
        metodo = ""
        x1 = y1 = x2 = y2 = None
        largura = altura = None

        try:
            imagem = ler_imagem(caminho_imagem)

            if imagem is None:
                raise ValueError("imagem_nao_abriu")

            altura, largura = imagem.shape[:2]
            x1, y1, x2, y2, metodo, erro_metodo = detectar_caixa(imagem)

            if erro_metodo:
                status = "fallback"
                erro = erro_metodo

            recorte = imagem[y1:y2, x1:x2].copy()

            anotada = imagem.copy()
            cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 255, 255), 10)
            cv2.putText(
                anotada,
                classe,
                (max(10, x1), max(40, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                4,
                cv2.LINE_AA,
            )

            if not salvar_imagem(caminho_recorte, recorte):
                raise ValueError("falha_ao_salvar_recorte")

            if not salvar_imagem(caminho_anotada, anotada):
                raise ValueError("falha_ao_salvar_anotada")

        except Exception as e:
            status = "erro"
            erro = str(e)

        registros.append({
            "classe": classe,
            "classe_yolo": CLASSE_PARA_INDICE.get(classe),
            "caminho_relativo_original": linha["caminho_relativo_original"],
            "arquivo_copiado": str(caminho_imagem),
            "nome_copiado": nome_copiado,
            "largura": largura,
            "altura": altura,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "metodo": metodo,
            "status_caixa": status,
            "erro": erro,
            "arquivo_recortado": caminho_relativo(caminho_recorte) if caminho_recorte.exists() else "",
            "arquivo_anotado": caminho_relativo(caminho_anotada) if caminho_anotada.exists() else "",
        })

    relatorio = pd.DataFrame(registros)
    caminho_saida = PASTA_TABELAS / "caixas_automaticas.csv"
    relatorio.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    resumo = (
        relatorio.groupby(["classe", "status_caixa"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["classe", "status_caixa"])
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(resumo.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {caminho_saida}")
    print(f"- {PASTA_RECORTE}")
    print(f"- {PASTA_CONFERENCIA}")
    print()
    print("Proximo passo:")
    print("python scripts\\09_conferir_caixas_automaticas.py")


if __name__ == "__main__":
    main()
