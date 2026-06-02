from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# SCRIPT 08B - GERAR CAIXAS AUTOMATICAS PARA PILOTO/TESTE 2
# ------------------------------------------------------------
# Objetivo:
# - Processar imagens com prefixo Piloto__ e TESTE_2__
# - Usar uma regra diferente da usada em Micro-ondas
# - Salvar recortes, imagens com caixas e relatorio CSV separado
#
# Estas fotos sao mais "close-up": a semente pode ocupar boa parte da imagem.
# Por isso o detector aceita caixas maiores e tenta unir varias manchas da
# semente antes de escolher a caixa final.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_RECORTE = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_CONFERENCIA = PASTA_PROJETO / "saidas" / "conferencia_caixas" / "imagens"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_PARA_INDICE = {"nao_contaminada": 0, "contaminada": 1}
PREFIXOS_ALVO = ("Piloto__", "TESTE_2__")
ORIGEM_DETECTOR = "piloto_teste2"

TAMANHO_PROCESSAMENTO = 900
MARGEM_CAIXA = 0.16
PADDING_FALLBACK = 0.08
ESCALA_TEXTO_CAIXA = 2.4
ESPESSURA_TEXTO_CAIXA = 7

DISTANCIAS_FUNDO_LAB = [16, 22, 30]
AREA_MINIMA_CONTORNO = 0.004
AREA_MINIMA_CAIXA = 0.030
AREA_MAXIMA_CAIXA = 0.90
ASPECT_RATIO_MINIMO = 0.20
ASPECT_RATIO_MAXIMO = 5.00
EXTENT_MINIMO = 0.08

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


def tamanho_kernel_impar(valor: int) -> int:
    valor = max(3, int(valor))
    return valor if valor % 2 == 1 else valor + 1


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
    largura_borda = max(5, min(35, altura // 7, largura // 7))

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


def mascara_hsv_nao_azul(imagem):
    """
    Alternativa simples: remove fundo muito azulado e muito claro.
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    fundo_azul = (h >= 80) & (h <= 150) & (s >= 25)
    muito_claro = (s <= 35) & (v >= 230)
    mascara = (~fundo_azul) & (~muito_claro) & (v <= 245)
    return mascara.astype(np.uint8) * 255


def limpar_mascara_closeup(mascara, largura: int, altura: int):
    """
    Junta manchas internas da semente para evitar caixas minusculas.
    """
    menor_lado = min(largura, altura)
    k_abertura = tamanho_kernel_impar(max(5, menor_lado * 0.010))
    k_fechamento = tamanho_kernel_impar(max(35, menor_lado * 0.060))
    k_dilatacao = tamanho_kernel_impar(max(5, menor_lado * 0.012))

    mascara = cv2.medianBlur(mascara, 5)
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        np.ones((k_abertura, k_abertura), np.uint8),
        iterations=1,
    )
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        np.ones((k_fechamento, k_fechamento), np.uint8),
        iterations=2,
    )
    mascara = cv2.dilate(
        mascara,
        np.ones((k_dilatacao, k_dilatacao), np.uint8),
        iterations=1,
    )
    return mascara


def mascaras_closeup(imagem):
    mascaras = []

    for distancia in DISTANCIAS_FUNDO_LAB:
        mascaras.append((
            f"closeup_fundo_lab_{distancia}",
            mascara_distancia_fundo_lab(imagem, distancia),
        ))

    mascaras.append(("closeup_hsv_nao_azul", mascara_hsv_nao_azul(imagem)))
    return mascaras


def avaliar_caixa(x, y, w, h, area, metodo: str, largura: int, altura: int):
    area_imagem = largura * altura
    area_caixa = w * h
    proporcao_area = area / area_imagem
    proporcao_caixa = area_caixa / area_imagem

    if proporcao_area < AREA_MINIMA_CONTORNO:
        return None

    if proporcao_caixa < AREA_MINIMA_CAIXA:
        return None

    if proporcao_caixa > AREA_MAXIMA_CAIXA:
        return None

    aspect_ratio = w / max(h, 1)

    if aspect_ratio < ASPECT_RATIO_MINIMO or aspect_ratio > ASPECT_RATIO_MAXIMO:
        return None

    extent = area / max(area_caixa, 1)

    if extent < EXTENT_MINIMO:
        return None

    penalidade_caixa_grande = 0.55 if proporcao_caixa > 0.72 else 1.0
    bonus_uniao = 1.25 if "componentes_unidos" in metodo else 1.0

    # Para close-up, uma caixa um pouco maior e melhor que uma caixa minuscula.
    score = area * max(extent, 0.10) * penalidade_caixa_grande * bonus_uniao

    return {
        "metodo": metodo,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
    }


def coletar_candidatos_por_mascara(mascara, nome_mascara: str, largura: int, altura: int):
    mascara = limpar_mascara_closeup(mascara, largura, altura)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidatos = []
    area_imagem = largura * altura
    contornos_relevantes = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)

        candidato = avaliar_caixa(x, y, w, h, area, nome_mascara, largura, altura)

        if candidato is not None:
            candidatos.append(candidato)

        if area / area_imagem >= (AREA_MINIMA_CONTORNO / 2):
            contornos_relevantes.append(contorno)

    if len(contornos_relevantes) >= 2:
        pontos = np.vstack(contornos_relevantes)
        x, y, w, h = cv2.boundingRect(pontos)
        area_somada = sum(cv2.contourArea(contorno) for contorno in contornos_relevantes)
        candidato = avaliar_caixa(
            x,
            y,
            w,
            h,
            area_somada,
            f"{nome_mascara}_componentes_unidos",
            largura,
            altura,
        )

        if candidato is not None:
            candidatos.append(candidato)

    return candidatos


def detectar_caixa_closeup(imagem):
    """
    Detecta caixas para Piloto/TESTE 2.

    A regra evita caixas muito pequenas dentro da semente e aceita caixas
    maiores, porque essas fotos geralmente sao mais aproximadas.
    """
    altura_original, largura_original = imagem.shape[:2]
    menor, escala = redimensionar_para_processamento(imagem)
    altura_menor, largura_menor = menor.shape[:2]
    candidatos = []

    for nome_mascara, mascara in mascaras_closeup(menor):
        candidatos.extend(
            coletar_candidatos_por_mascara(
                mascara,
                nome_mascara,
                largura_menor,
                altura_menor,
            )
        )

    if not candidatos:
        x1, y1, x2, y2 = caixa_fallback(largura_original, altura_original)
        return x1, y1, x2, y2, "closeup_fallback", "sem_contorno_plausivel"

    melhor = sorted(candidatos, key=lambda item: item["score"], reverse=True)[0]

    x1 = int(melhor["x"] / escala)
    y1 = int(melhor["y"] / escala)
    x2 = int((melhor["x"] + melhor["w"]) / escala)
    y2 = int((melhor["y"] + melhor["h"]) / escala)

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
    print("GERANDO CAIXAS AUTOMATICAS - PILOTO/TESTE 2")
    print("=" * 60)

    caminho_relatorio = PASTA_TABELAS / "relatorio_copia_dataset_binario.csv"

    if not caminho_relatorio.exists():
        print("ERRO: relatorio_copia_dataset_binario.csv nao encontrado.")
        print(caminho_relatorio)
        return

    PASTA_RECORTE.mkdir(parents=True, exist_ok=True)
    PASTA_CONFERENCIA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_relatorio)
    nome_copiado = df["nome_copiado"].astype(str)
    mascara_origem = pd.Series(False, index=df.index)

    for prefixo in PREFIXOS_ALVO:
        mascara_origem = mascara_origem | nome_copiado.str.startswith(prefixo)

    df = df[
        (df["status_copia"] == "ok")
        & (df["classe"].isin(CLASSES))
        & mascara_origem
    ].copy()

    print(f"Imagens Piloto/TESTE 2 para processar: {len(df)}")
    print()

    registros = []

    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Detectando caixas Piloto/TESTE 2"):
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
            x1, y1, x2, y2, metodo, erro_metodo = detectar_caixa_closeup(imagem)

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
    caminho_saida = PASTA_TABELAS / "caixas_piloto_teste2.csv"
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
        print("Nenhuma imagem Piloto/TESTE 2 encontrada.")

    print()
    print("Arquivos gerados:")
    print(f"- {caminho_saida}")
    print(f"- {PASTA_RECORTE}")
    print(f"- {PASTA_CONFERENCIA}")
    print()
    print("Proximo passo:")
    print("python scripts\\08c_juntar_caixas_automaticas.py")


if __name__ == "__main__":
    main()
