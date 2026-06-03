from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 11 - MARCAR AJUSTES MANUAIS DE CAIXAS
# ------------------------------------------------------------
# Objetivo:
# - Abrir imagens filtradas pelo nome
# - Permitir desenhar a caixa correta com o mouse
# - Salvar somente os ajustes manuais em CSV separado
#
# Este script NAO altera as imagens originais.
# Este script NAO altera caixas_automaticas.csv diretamente.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_CAIXAS_TABELAS = PASTA_TABELAS / "05_caixas_yolo"
CAMINHO_CAIXAS = PASTA_CAIXAS_TABELAS / "caixas_automaticas.csv"
CAMINHO_AJUSTES = PASTA_CAIXAS_TABELAS / "caixas_ajustes_manuais.csv"

TAMANHO_MAXIMO_TELA = 1100


def ler_imagem(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def redimensionar_para_tela(imagem):
    altura, largura = imagem.shape[:2]
    maior_lado = max(altura, largura)

    if maior_lado <= TAMANHO_MAXIMO_TELA:
        return imagem.copy(), 1.0

    escala = TAMANHO_MAXIMO_TELA / maior_lado
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    menor = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return menor, escala


def desenhar_caixa_atual(imagem, linha):
    anotada = imagem.copy()

    try:
        x1 = int(float(linha["x1"]))
        y1 = int(float(linha["y1"]))
        x2 = int(float(linha["x2"]))
        y2 = int(float(linha["y2"]))
    except Exception:
        return anotada

    cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 255, 255), 8)
    return anotada


def carregar_ajustes_existentes():
    PASTA_CAIXAS_TABELAS.mkdir(parents=True, exist_ok=True)

    if CAMINHO_AJUSTES.exists():
        return pd.read_csv(CAMINHO_AJUSTES)

    return pd.DataFrame(columns=[
        "nome_copiado",
        "x1_manual",
        "y1_manual",
        "x2_manual",
        "y2_manual",
        "status_manual",
        "observacao_manual",
    ])


def salvar_ajuste(df_ajustes, registro):
    df_ajustes = df_ajustes[df_ajustes["nome_copiado"] != registro["nome_copiado"]].copy()
    df_ajustes = pd.concat([df_ajustes, pd.DataFrame([registro])], ignore_index=True)
    df_ajustes = df_ajustes.sort_values("nome_copiado")
    df_ajustes.to_csv(CAMINHO_AJUSTES, index=False, encoding="utf-8-sig")
    return df_ajustes


def selecionar_roi(imagem_tela, titulo):
    """
    Abre uma janela para desenhar a caixa.

    Controles do OpenCV:
    - arraste o mouse para marcar a caixa
    - Enter ou Espaco confirma
    - C cancela a imagem atual
    """
    roi = cv2.selectROI(titulo, imagem_tela, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(titulo)
    return roi


def main():
    parser = ArgumentParser(description="Marcar ajustes manuais de caixas.")
    parser.add_argument(
        "--filtro",
        default="TESTE_2__T6__",
        help="Trecho do nome_copiado que sera ajustado. Padrao: TESTE_2__T6__",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MARCANDO AJUSTES MANUAIS DE CAIXAS")
    print("=" * 60)

    if not CAMINHO_CAIXAS.exists():
        print("ERRO: caixas_automaticas.csv nao encontrado.")
        print("Execute primeiro os scripts 08, 09 e 10.")
        return

    df = pd.read_csv(CAMINHO_CAIXAS)
    df = df[df["nome_copiado"].astype(str).str.contains(args.filtro, regex=False)].copy()
    df = df.sort_values("nome_copiado")

    if len(df) == 0:
        print(f"Nenhuma imagem encontrada com filtro: {args.filtro}")
        return

    df_ajustes = carregar_ajustes_existentes()

    print(f"Imagens encontradas: {len(df)}")
    print()
    print("Para cada imagem:")
    print("- arraste o mouse para marcar a caixa correta")
    print("- pressione Enter ou Espaco para confirmar")
    print("- pressione C para pular/cancelar a imagem atual")
    print("- feche a janela se quiser parar")
    print()

    for indice, (_, linha) in enumerate(df.iterrows(), start=1):
        nome_copiado = str(linha["nome_copiado"])
        caminho_imagem = Path(str(linha["arquivo_copiado"]))

        imagem = ler_imagem(caminho_imagem)

        if imagem is None:
            print(f"[{indice}/{len(df)}] ERRO ao abrir: {nome_copiado}")
            continue

        imagem_com_caixa = desenhar_caixa_atual(imagem, linha)
        imagem_tela, escala = redimensionar_para_tela(imagem_com_caixa)

        titulo = f"{indice}/{len(df)} - {nome_copiado}"
        print(f"[{indice}/{len(df)}] Ajustando: {nome_copiado}")
        x, y, w, h = selecionar_roi(imagem_tela, titulo)

        if w <= 0 or h <= 0:
            print("  pulado")
            continue

        x1 = int(x / escala)
        y1 = int(y / escala)
        x2 = int((x + w) / escala)
        y2 = int((y + h) / escala)

        registro = {
            "nome_copiado": nome_copiado,
            "x1_manual": x1,
            "y1_manual": y1,
            "x2_manual": x2,
            "y2_manual": y2,
            "status_manual": "ok",
            "observacao_manual": f"ajuste_manual_filtro_{args.filtro}",
        }
        df_ajustes = salvar_ajuste(df_ajustes, registro)

        print(f"  salvo: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    print()
    print("Arquivo gerado/atualizado:")
    print(f"- {CAMINHO_AJUSTES}")


if __name__ == "__main__":
    main()



