from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# SCRIPT 08 - GERAR CAIXAS PARA MICRO-ONDAS
# ------------------------------------------------------------
# Objetivo:
# - Processar somente imagens com prefixo Micro-ondas__
# - Detectar automaticamente a regiao provavel da semente
# - Salvar recortes, imagens com caixas e relatorio CSV separado
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
PREFIXO_ALVO = "Micro-ondas__"
ORIGEM_DETECTOR = "microondas"

TAMANHO_PROCESSAMENTO = 900
MARGEM_CAIXA = 0.28
ESCALA_TEXTO_CAIXA = 2.4
ESPESSURA_TEXTO_CAIXA = 7
AREA_MINIMA_PROPORCAO = 0.00015
AREA_MAXIMA_PROPORCAO = 0.10
AREA_MAXIMA_CAIXA_PROPORCAO = 0.18
ASPECT_RATIO_MINIMO = 0.30
ASPECT_RATIO_MAXIMO = 3.20
EXTENT_MINIMO = 0.12
CIRCULARIDADE_MINIMA = 0.08
PADDING_FALLBACK = 0.08
DISTANCIA_MINIMA_FUNDO_LAB = 35

COLUNAS_RELATORIO = [
    "origem_detector",
    "classe",
    "classe_yolo",
    "caminho_relativo_original",
    "arquivo_copiado",
    "nome_copiado",
    "largura",
    "altura",
    "x1",
    "y1",
    "x2",
    "y2",
    "metodo",
    "status_caixa",
    "erro",
    "arquivo_recortado",
    "arquivo_anotado",
]

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


def cor_fundo_lab_por_bordas(imagem):
    altura, largura = imagem.shape[:2]
    largura_borda = max(5, min(30, altura // 8, largura // 8))

    borda = np.concatenate([
        imagem[:largura_borda, :, :].reshape(-1, 3),
        imagem[-largura_borda:, :, :].reshape(-1, 3),
        imagem[:, :largura_borda, :].reshape(-1, 3),
        imagem[:, -largura_borda:, :].reshape(-1, 3),
    ])
    borda_lab = cv2.cvtColor(
        borda.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3)

    return np.median(borda_lab, axis=0)


def mascara_distancia_fundo_lab(imagem, distancia_minima):
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    cor_fundo_lab = cor_fundo_lab_por_bordas(imagem)
    distancia_fundo = np.linalg.norm(
        lab.astype(float) - cor_fundo_lab.reshape(1, 1, 3),
        axis=2,
    )
    return (distancia_fundo >= distancia_minima).astype(np.uint8) * 255


def mascara_fundo_estimado(imagem):
    """
    Segmenta objetos diferentes do fundo azul estimado pelas bordas.

    Nas imagens Micro-ondas, a etiqueta branca compete com a semente.
    Por isso removemos regioes claras de baixa saturacao.
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mascara_distancia = mascara_distancia_fundo_lab(imagem, DISTANCIA_MINIMA_FUNDO_LAB)

    etiqueta_clara = (s <= 80) & (v >= 145)
    muito_claro = v >= 220
    objeto_diferente_do_fundo = mascara_distancia > 0
    mascara = objeto_diferente_do_fundo & (~etiqueta_clara) & (~muito_claro)
    return mascara.astype(np.uint8) * 255


def mascaras_microondas(imagem):
    """
    Cria mascaras que favorecem sementes e penalizam fundo azul/etiqueta.
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    fundo_azul = (h >= 80) & (h <= 150) & (s >= 35)
    etiqueta_branca = (s <= 70) & (v >= 155)
    muito_claro = v >= 225

    nao_fundo = ~fundo_azul
    nao_etiqueta = ~etiqueta_branca

    mascara_escura = nao_fundo & nao_etiqueta & (v <= 185)
    mascara_media = nao_fundo & nao_etiqueta & (~muito_claro) & (v <= 215)

    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    canal_b = lab[:, :, 2]
    mascara_marrom = nao_fundo & nao_etiqueta & (canal_b >= 120) & (v <= 220)

    return [
        ("fundo_estimado_lab", mascara_fundo_estimado(imagem)),
        ("hsv_escuro_nao_azul", mascara_escura.astype(np.uint8) * 255),
        ("hsv_medio_nao_azul", mascara_media.astype(np.uint8) * 255),
        ("lab_marrom_nao_azul", mascara_marrom.astype(np.uint8) * 255),
    ]


def avaliar_contorno(contorno, nome_mascara: str, largura: int, altura: int):
    area = cv2.contourArea(contorno)
    area_imagem = largura * altura
    proporcao_area = area / area_imagem

    if proporcao_area < AREA_MINIMA_PROPORCAO:
        return None

    if proporcao_area > AREA_MAXIMA_PROPORCAO:
        return None

    x, y, w, h = cv2.boundingRect(contorno)
    area_caixa = w * h
    proporcao_caixa = area_caixa / area_imagem

    if proporcao_caixa > AREA_MAXIMA_CAIXA_PROPORCAO:
        return None

    aspect_ratio = w / max(h, 1)

    if aspect_ratio < ASPECT_RATIO_MINIMO or aspect_ratio > ASPECT_RATIO_MAXIMO:
        return None

    extent = area / max(area_caixa, 1)

    if extent < EXTENT_MINIMO:
        return None

    perimetro = cv2.arcLength(contorno, True)
    circularidade = 0.0

    if perimetro > 0:
        circularidade = (4 * np.pi * area) / (perimetro * perimetro)

    if circularidade < CIRCULARIDADE_MINIMA:
        return None

    toca_borda = x <= 2 or y <= 2 or (x + w) >= (largura - 2) or (y + h) >= (altura - 2)
    penalidade_borda = 0.30 if toca_borda else 1.0

    bonus_metodo = 1.6 if nome_mascara == "fundo_estimado_lab" else 1.0

    score = area * extent * max(circularidade, 0.05) * penalidade_borda * bonus_metodo

    return {
        "metodo": nome_mascara,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
    }


def coletar_candidatos_por_mascara(mascara, nome_mascara: str, largura: int, altura: int):
    kernel_pequeno = np.ones((5, 5), np.uint8)
    kernel_medio = np.ones((9, 9), np.uint8)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_pequeno, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_medio, iterations=2)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidatos = []

    for contorno in contornos:
        candidato = avaliar_contorno(contorno, nome_mascara, largura, altura)

        if candidato is not None:
            candidatos.append(candidato)

    return candidatos


def detectar_caixa(imagem):
    """
    Detecta a regiao mais provavel da semente em imagens Micro-ondas.
    """
    altura_original, largura_original = imagem.shape[:2]
    menor, escala = redimensionar_para_processamento(imagem)
    altura_menor, largura_menor = menor.shape[:2]
    candidatos = []

    for nome_mascara, mascara in mascaras_microondas(menor):
        candidatos.extend(
            coletar_candidatos_por_mascara(
                mascara,
                nome_mascara,
                largura_menor,
                altura_menor,
            )
        )

    if not candidatos:
        gray = cv2.cvtColor(menor, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        _, mascara_otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        candidatos.extend(
            coletar_candidatos_por_mascara(
                mascara_otsu,
                "otsu_secundario",
                largura_menor,
                altura_menor,
            )
        )

    if not candidatos:
        x1, y1, x2, y2 = caixa_fallback(largura_original, altura_original)
        return x1, y1, x2, y2, "fallback", "sem_contorno_plausivel"

    melhor = sorted(candidatos, key=lambda item: item["score"], reverse=True)[0]

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
    print("GERANDO CAIXAS - MICRO-ONDAS")
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
        & (df["nome_copiado"].astype(str).str.startswith(PREFIXO_ALVO))
    ].copy()

    print(f"Imagens Micro-ondas para processar: {len(df)}")
    print()

    registros = []

    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Detectando caixas Micro-ondas"):
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
                ESCALA_TEXTO_CAIXA,
                (0, 255, 255),
                ESPESSURA_TEXTO_CAIXA,
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
            "origem_detector": ORIGEM_DETECTOR,
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

    relatorio = pd.DataFrame(registros, columns=COLUNAS_RELATORIO)
    caminho_saida = PASTA_TABELAS / "caixas_microondas.csv"
    relatorio.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    if len(relatorio):
        resumo = (
            relatorio.groupby(["classe", "status_caixa"])
            .size()
            .reset_index(name="quantidade")
            .sort_values(["classe", "status_caixa"])
        )
        print(resumo.to_string(index=False))
    else:
        print("Nenhuma imagem Micro-ondas encontrada.")

    print()
    print("Arquivos gerados:")
    print(f"- {caminho_saida}")
    print(f"- {PASTA_RECORTE}")
    print(f"- {PASTA_CONFERENCIA}")
    print()
    print("Proximos passos:")
    print("python scripts\\08b_gerar_caixas_piloto_teste2.py")
    print("python scripts\\08c_juntar_caixas_automaticas.py")


if __name__ == "__main__":
    main()
