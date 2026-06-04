from pathlib import Path
import argparse
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


# ============================================================
# SCRIPT 22 - EXTRAIR ATRIBUTOS VISUAIS DOS RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Reusar o split original de treino/validacao/teste
# - Extrair atributos visuais interpretaveis dos recortes
# - Nao usar origem, tratamento, pasta ou outros metadados como features
#
# Este script nao treina modelos e nao altera imagens.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_CLASSICOS = PASTA_TABELAS / "06_modelos" / "classicos"

CAMINHO_SPLIT = PASTA_DATASET_TABELAS / "divisao_treino_validacao_teste.csv"
CAMINHO_ATRIBUTOS = PASTA_CLASSICOS / "atributos_visuais_recortes.csv"
CAMINHO_RESUMO = PASTA_CLASSICOS / "resumo_atributos_visuais_recortes.csv"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
HIST_BINS_HSV = 16
LBP_BINS = 256

COLUNAS_AUDITORIA_PRIORITARIAS = [
    "nome_arquivo",
    "split",
    "classe",
    "alvo",
    "caminho_imagem",
    "caminho_recorte",
    "status_atributos",
    "erro_atributos",
]

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def calcular_workers_padrao() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai atributos visuais dos recortes usando o split original."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=calcular_workers_padrao(),
        help=(
            "Numero de processos para extracao em CPU. "
            f"Padrao: {calcular_workers_padrao()}."
        ),
    )
    return parser


def inicializar_worker():
    cv2.setNumThreads(1)


def ler_split() -> pd.DataFrame:
    if not CAMINHO_SPLIT.exists():
        raise FileNotFoundError(f"Split original nao encontrado: {CAMINHO_SPLIT}")

    df = pd.read_csv(CAMINHO_SPLIT)
    colunas_obrigatorias = ["nome_arquivo", "split", "classe"]
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes no split: {faltantes}")

    df = df[df["classe"].isin(CLASSES)].copy()
    if "alvo" not in df.columns:
        df["alvo"] = df["classe"].map(CLASSE_PARA_INDICE)

    return df.reset_index(drop=True)


def caminho_recorte_para_linha(linha) -> Path:
    classe = str(linha["classe"])
    nome_arquivo = str(linha["nome_arquivo"])
    return PASTA_DATASET / classe / nome_arquivo


def carregar_imagem_rgb(caminho: Path) -> np.ndarray:
    with Image.open(caminho) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.asarray(img)


def componente_principal(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8)
    if num_labels <= 1:
        return np.zeros_like(mask_uint8, dtype=bool), {
            "area": 0,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    indice = int(np.argmax(areas) + 1)
    x = int(stats[indice, cv2.CC_STAT_LEFT])
    y = int(stats[indice, cv2.CC_STAT_TOP])
    w = int(stats[indice, cv2.CC_STAT_WIDTH])
    h = int(stats[indice, cv2.CC_STAT_HEIGHT])
    area = int(stats[indice, cv2.CC_STAT_AREA])
    return labels == indice, {
        "area": area,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
    }


def pontuar_mascara(mask: np.ndarray, info: dict, altura: int, largura: int) -> float:
    area_total = max(altura * largura, 1)
    area_frac = info["area"] / area_total
    if area_frac <= 0.01 or area_frac >= 0.98:
        return -1.0

    centro_x = info["bbox_x"] + info["bbox_w"] / 2
    centro_y = info["bbox_y"] + info["bbox_h"] / 2
    dist_centro = math.hypot(
        (centro_x / max(largura, 1)) - 0.5,
        (centro_y / max(altura, 1)) - 0.5,
    )
    toca_borda = (
        info["bbox_x"] <= 1
        or info["bbox_y"] <= 1
        or info["bbox_x"] + info["bbox_w"] >= largura - 1
        or info["bbox_y"] + info["bbox_h"] >= altura - 1
    )
    penalidade_borda = 0.15 if toca_borda else 0.0
    preferencia_area = 1.0 - abs(area_frac - 0.45)
    return preferencia_area - dist_centro - penalidade_borda


def criar_mascara_semente(rgb: np.ndarray) -> np.ndarray:
    altura, largura = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sat_blur = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

    candidatos = []
    for imagem_base in [gray_blur, sat_blur]:
        _, binaria = cv2.threshold(
            imagem_base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        for mask in [binaria > 0, binaria == 0]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_limpa = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask_limpa = cv2.morphologyEx(mask_limpa, cv2.MORPH_CLOSE, kernel)
            principal, info = componente_principal(mask_limpa > 0)
            score = pontuar_mascara(principal, info, altura, largura)
            candidatos.append((score, principal))

    melhor_score, melhor_mask = max(candidatos, key=lambda item: item[0])
    if melhor_score < 0 or not melhor_mask.any():
        return np.ones((altura, largura), dtype=bool)
    return melhor_mask.astype(bool)


def valores_mascarados(imagem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valores = imagem[mask]
    if valores.size == 0:
        return imagem.reshape(-1, imagem.shape[-1])
    return valores


def adicionar_estatisticas_canais(
    atributos: dict,
    prefixo: str,
    imagem: np.ndarray,
    mask: np.ndarray,
    nomes_canais: list[str],
):
    valores = valores_mascarados(imagem, mask).astype(float)
    percentis = np.percentile(valores, [5, 25, 50, 75, 95], axis=0)

    for indice, nome_canal in enumerate(nomes_canais):
        canal = valores[:, indice]
        atributos[f"{prefixo}_mean_{nome_canal}"] = float(np.mean(canal))
        atributos[f"{prefixo}_std_{nome_canal}"] = float(np.std(canal))
        atributos[f"{prefixo}_min_{nome_canal}"] = float(np.min(canal))
        atributos[f"{prefixo}_max_{nome_canal}"] = float(np.max(canal))
        for p_indice, p_nome in enumerate(["p05", "p25", "p50", "p75", "p95"]):
            atributos[f"{prefixo}_{p_nome}_{nome_canal}"] = float(
                percentis[p_indice, indice]
            )


def adicionar_histogramas_hsv(atributos: dict, hsv: np.ndarray, mask: np.ndarray):
    ranges = {
        "h": (0, 180),
        "s": (0, 256),
        "v": (0, 256),
    }
    canais = {"h": 0, "s": 1, "v": 2}

    for nome, canal in canais.items():
        valores = hsv[:, :, canal][mask]
        if valores.size == 0:
            valores = hsv[:, :, canal].reshape(-1)
        hist, _ = np.histogram(
            valores,
            bins=HIST_BINS_HSV,
            range=ranges[nome],
        )
        hist = hist.astype(float)
        hist = hist / max(hist.sum(), 1.0)
        for indice, valor in enumerate(hist):
            atributos[f"hsv_hist_{nome}_{indice:02d}"] = float(valor)


def adicionar_brilho_contraste(
    atributos: dict,
    gray: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
):
    brilho = hsv[:, :, 2][mask]
    gray_valores = gray[mask]
    if brilho.size == 0:
        brilho = hsv[:, :, 2].reshape(-1)
    if gray_valores.size == 0:
        gray_valores = gray.reshape(-1)

    percentis = np.percentile(brilho, [1, 5, 25, 50, 75, 95, 99])
    atributos["brilho_mean"] = float(np.mean(brilho))
    atributos["brilho_std"] = float(np.std(brilho))
    for valor, nome in zip(
        percentis,
        ["p01", "p05", "p25", "p50", "p75", "p95", "p99"],
    ):
        atributos[f"brilho_{nome}"] = float(valor)

    atributos["contraste_gray_std"] = float(np.std(gray_valores))
    atributos["contraste_gray_p95_p05"] = float(
        np.percentile(gray_valores, 95) - np.percentile(gray_valores, 5)
    )
    atributos["contraste_gray_p75_p25"] = float(
        np.percentile(gray_valores, 75) - np.percentile(gray_valores, 25)
    )


def obter_contorno_principal(mask: np.ndarray):
    contornos, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def adicionar_forma(atributos: dict, mask: np.ndarray):
    altura, largura = mask.shape
    area_total = max(altura * largura, 1)
    area_mask = int(mask.sum())
    atributos["mask_area_px"] = area_mask
    atributos["mask_area_frac"] = float(area_mask / area_total)

    contorno = obter_contorno_principal(mask)
    if contorno is None or len(contorno) < 3:
        valores_padrao = {
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": largura,
            "bbox_h": altura,
            "bbox_area_frac": 1.0,
            "bbox_aspect_ratio": largura / max(altura, 1),
            "contour_area_px": float(area_mask),
            "contour_perimeter_px": 0.0,
            "circularity": 0.0,
            "extent": float(area_mask / area_total),
            "solidity": 0.0,
        }
        atributos.update(valores_padrao)
        for indice in range(7):
            atributos[f"hu_moment_log_abs_{indice + 1}"] = 0.0
        return

    x, y, w, h = cv2.boundingRect(contorno)
    area_contorno = float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    area_bbox = max(w * h, 1)
    hull = cv2.convexHull(contorno)
    area_hull = float(cv2.contourArea(hull))
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()

    atributos["bbox_x"] = int(x)
    atributos["bbox_y"] = int(y)
    atributos["bbox_w"] = int(w)
    atributos["bbox_h"] = int(h)
    atributos["bbox_area_frac"] = float(area_bbox / area_total)
    atributos["bbox_aspect_ratio"] = float(w / max(h, 1))
    atributos["contour_area_px"] = area_contorno
    atributos["contour_perimeter_px"] = perimetro
    atributos["circularity"] = float(
        (4 * math.pi * area_contorno) / max(perimetro * perimetro, 1e-9)
    )
    atributos["extent"] = float(area_contorno / area_bbox)
    atributos["solidity"] = float(area_contorno / max(area_hull, 1e-9))

    for indice, valor in enumerate(hu):
        sinal = -1.0 if valor < 0 else 1.0
        atributos[f"hu_moment_log_abs_{indice + 1}"] = float(
            sinal * math.log10(abs(float(valor)) + 1e-30)
        )


def calcular_lbp(gray: np.ndarray) -> np.ndarray:
    centro = gray[1:-1, 1:-1]
    vizinhos = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]

    lbp = np.zeros_like(centro, dtype=np.uint8)
    for bit, vizinho in enumerate(vizinhos):
        lbp |= ((vizinho >= centro).astype(np.uint8) << bit)
    return lbp


def adicionar_textura(atributos: dict, gray: np.ndarray, mask: np.ndarray):
    if mask.any():
        mediana = float(np.median(gray[mask]))
    else:
        mediana = float(np.median(gray))

    gray_mascarado = np.where(mask, gray, mediana).astype(np.float32)
    laplaciano = cv2.Laplacian(gray_mascarado, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray_mascarado, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_mascarado, cv2.CV_32F, 0, 1, ksize=3)
    energia_sobel = sobel_x ** 2 + sobel_y ** 2

    valores_lap = laplaciano[mask] if mask.any() else laplaciano.reshape(-1)
    valores_sobel = energia_sobel[mask] if mask.any() else energia_sobel.reshape(-1)
    atributos["texture_laplacian_var"] = float(np.var(valores_lap))
    atributos["texture_sobel_energy_mean"] = float(np.mean(valores_sobel))
    atributos["texture_sobel_energy_std"] = float(np.std(valores_sobel))

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        hist = np.zeros(LBP_BINS, dtype=float)
        hist[0] = 1.0
    else:
        lbp = calcular_lbp(gray)
        mask_interna = mask[1:-1, 1:-1]
        codigos = lbp[mask_interna] if mask_interna.any() else lbp.reshape(-1)
        hist, _ = np.histogram(codigos, bins=LBP_BINS, range=(0, LBP_BINS))
        hist = hist.astype(float)
        hist = hist / max(hist.sum(), 1.0)

    for indice, valor in enumerate(hist):
        atributos[f"lbp_hist_{indice:03d}"] = float(valor)


def extrair_atributos_imagem(caminho: Path) -> dict:
    rgb = carregar_imagem_rgb(caminho)
    altura, largura = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = criar_mascara_semente(rgb)

    atributos = {
        "largura_recorte": int(largura),
        "altura_recorte": int(altura),
        "pixels_recorte": int(largura * altura),
    }

    adicionar_estatisticas_canais(atributos, "rgb", rgb, mask, ["r", "g", "b"])
    adicionar_estatisticas_canais(atributos, "hsv", hsv, mask, ["h", "s", "v"])
    adicionar_histogramas_hsv(atributos, hsv, mask)
    adicionar_brilho_contraste(atributos, gray, hsv, mask)
    adicionar_forma(atributos, mask)
    adicionar_textura(atributos, gray, mask)

    return atributos


def registro_base(linha, caminho_recorte: Path) -> dict:
    registro = dict(linha)
    registro["classe_real"] = str(linha["classe"])
    registro["alvo"] = int(linha["alvo"])
    registro["caminho_recorte"] = str(caminho_recorte.relative_to(PASTA_PROJETO))
    return registro


def processar_linha_worker(linha: dict) -> dict:
    caminho_recorte = caminho_recorte_para_linha(linha)
    registro = registro_base(linha, caminho_recorte)

    if not caminho_recorte.exists():
        registro["status_atributos"] = "recorte_ausente"
        registro["erro_atributos"] = str(caminho_recorte)
        return registro

    try:
        atributos = extrair_atributos_imagem(caminho_recorte)
        registro.update(atributos)
        registro["status_atributos"] = "ok"
        registro["erro_atributos"] = ""
    except Exception as erro:  # noqa: BLE001 - registra falhas por imagem sem parar tudo.
        registro["status_atributos"] = "erro_extracao"
        registro["erro_atributos"] = repr(erro)

    return registro


def extrair_atributos(df_split: pd.DataFrame, workers: int) -> pd.DataFrame:
    linhas = df_split.to_dict("records")
    workers = max(1, int(workers))

    if workers == 1:
        inicializar_worker()
        registros = [
            processar_linha_worker(linha)
            for linha in tqdm(
                linhas,
                total=len(linhas),
                desc="Extraindo atributos dos recortes",
            )
        ]
        return pd.DataFrame(registros)

    chunksize = max(1, len(linhas) // (workers * 4)) if linhas else 1
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=inicializar_worker,
    ) as executor:
        registros = list(
            tqdm(
                executor.map(
                    processar_linha_worker,
                    linhas,
                    chunksize=chunksize,
                ),
                total=len(linhas),
                desc="Extraindo atributos dos recortes",
            )
        )
    return pd.DataFrame(registros)


def ordenar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    auditoria = [
        coluna for coluna in COLUNAS_AUDITORIA_PRIORITARIAS if coluna in df.columns
    ]
    restantes_auditoria = [
        coluna
        for coluna in df.columns
        if coluna not in auditoria and not coluna_e_atributo_visual(coluna)
    ]
    features = sorted(coluna for coluna in df.columns if coluna_e_atributo_visual(coluna))
    return df[auditoria + restantes_auditoria + features]


def coluna_e_atributo_visual(coluna: str) -> bool:
    prefixos = (
        "rgb_",
        "hsv_",
        "brilho_",
        "contraste_",
        "mask_",
        "bbox_",
        "contour_",
        "circularity",
        "extent",
        "solidity",
        "hu_moment_",
        "texture_",
        "lbp_",
        "largura_recorte",
        "altura_recorte",
        "pixels_recorte",
    )
    return coluna.startswith(prefixos)


def gerar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for coluna in ["mask_area_frac", "largura_recorte", "altura_recorte"]:
        if coluna not in df.columns:
            df[coluna] = np.nan

    agrupado = (
        df.groupby(["split", "classe", "status_atributos"], dropna=False)
        .agg(
            total=("nome_arquivo", "count"),
            mask_area_frac_media=("mask_area_frac", "mean"),
            mask_area_frac_mediana=("mask_area_frac", "median"),
            largura_recorte_mediana=("largura_recorte", "median"),
            altura_recorte_mediana=("altura_recorte", "median"),
        )
        .reset_index()
    )

    n_features = sum(coluna_e_atributo_visual(coluna) for coluna in df.columns)
    resumo_global = pd.DataFrame([
        {
            "split": "todos",
            "classe": "todas",
            "status_atributos": "resumo_global",
            "total": len(df),
            "mask_area_frac_media": df["mask_area_frac"].mean(),
            "mask_area_frac_mediana": df["mask_area_frac"].median(),
            "largura_recorte_mediana": df["largura_recorte"].median(),
            "altura_recorte_mediana": df["altura_recorte"].median(),
            "quantidade_features_visuais": n_features,
        }
    ])

    if "quantidade_features_visuais" not in agrupado.columns:
        agrupado["quantidade_features_visuais"] = n_features

    return pd.concat([agrupado, resumo_global], ignore_index=True)


def main():
    args = criar_parser().parse_args()
    workers = max(1, int(args.workers))

    print("=" * 60)
    print("EXTRAINDO ATRIBUTOS VISUAIS DOS RECORTES")
    print("=" * 60)

    PASTA_CLASSICOS.mkdir(parents=True, exist_ok=True)

    df_split = ler_split()
    print(f"Registros no split original: {len(df_split)}")
    print(f"Workers CPU: {workers}")
    print(pd.crosstab(df_split["split"], df_split["classe"]).to_string())

    atributos = extrair_atributos(df_split, workers=workers)
    atributos = ordenar_colunas(atributos)
    resumo = gerar_resumo(atributos)

    atributos.to_csv(CAMINHO_ATRIBUTOS, index=False, encoding="utf-8-sig")
    resumo.to_csv(CAMINHO_RESUMO, index=False, encoding="utf-8-sig")

    print()
    print("Resumo da extracao:")
    print(
        atributos.groupby(["split", "classe", "status_atributos"], dropna=False)
        .size()
        .reset_index(name="total")
        .to_string(index=False)
    )
    print()
    print("Arquivos gerados:")
    print(f"- {CAMINHO_ATRIBUTOS}")
    print(f"- {CAMINHO_RESUMO}")


if __name__ == "__main__":
    main()
