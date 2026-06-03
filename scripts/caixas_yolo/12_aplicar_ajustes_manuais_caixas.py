from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 12 - APLICAR AJUSTES MANUAIS DE CAIXAS
# ------------------------------------------------------------
# Objetivo:
# - Ler caixas_automaticas.csv
# - Aplicar coordenadas de caixas_ajustes_manuais.csv
# - Atualizar recortes e imagens de conferencia
#
# Este script preserva a rastreabilidade:
# - metodo vira ajuste_manual
# - status_caixa fica ok
# - erro fica vazio
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_CAIXAS_TABELAS = PASTA_TABELAS / "05_caixas_yolo"
PASTA_RECORTE = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_CONFERENCIA = PASTA_PROJETO / "saidas" / "conferencia_caixas" / "imagens"

CAMINHO_CAIXAS = PASTA_CAIXAS_TABELAS / "caixas_automaticas.csv"
CAMINHO_AJUSTES = PASTA_CAIXAS_TABELAS / "caixas_ajustes_manuais.csv"


def ler_imagem(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho: Path, imagem) -> bool:
    extensao = caminho.suffix or ".jpg"
    ok, buffer = cv2.imencode(extensao, imagem)

    if not ok:
        return False

    buffer.tofile(str(caminho))
    return True


def caminho_relativo(caminho: Path) -> str:
    return caminho.relative_to(PASTA_PROJETO).as_posix()


def limitar(valor, minimo, maximo):
    return max(minimo, min(int(valor), maximo))


def main():
    print("=" * 60)
    print("APLICANDO AJUSTES MANUAIS DE CAIXAS")
    print("=" * 60)

    if not CAMINHO_CAIXAS.exists():
        print("ERRO: caixas_automaticas.csv nao encontrado.")
        return

    if not CAMINHO_AJUSTES.exists():
        print("ERRO: caixas_ajustes_manuais.csv nao encontrado.")
        print("Execute primeiro: python scripts\\11_marcar_ajustes_manuais_caixas.py")
        return

    df = pd.read_csv(CAMINHO_CAIXAS)
    ajustes = pd.read_csv(CAMINHO_AJUSTES)
    ajustes = ajustes[ajustes["status_manual"] == "ok"].copy()

    if len(ajustes) == 0:
        print("Nenhum ajuste manual com status_manual == ok.")
        return

    ajustes_por_nome = ajustes.set_index("nome_copiado").to_dict(orient="index")
    quantidade_aplicada = 0
    erros = []

    for indice, linha in df.iterrows():
        nome_copiado = str(linha["nome_copiado"])

        if nome_copiado not in ajustes_por_nome:
            continue

        ajuste = ajustes_por_nome[nome_copiado]
        classe = str(linha["classe"])
        caminho_imagem = Path(str(linha["arquivo_copiado"]))
        imagem = ler_imagem(caminho_imagem)

        if imagem is None:
            erros.append((nome_copiado, "imagem_nao_abriu"))
            continue

        altura, largura = imagem.shape[:2]
        x1 = limitar(ajuste["x1_manual"], 0, largura - 1)
        y1 = limitar(ajuste["y1_manual"], 0, altura - 1)
        x2 = limitar(ajuste["x2_manual"], x1 + 1, largura)
        y2 = limitar(ajuste["y2_manual"], y1 + 1, altura)

        pasta_recorte_classe = PASTA_RECORTE / classe
        pasta_conf_classe = PASTA_CONFERENCIA / classe
        pasta_recorte_classe.mkdir(parents=True, exist_ok=True)
        pasta_conf_classe.mkdir(parents=True, exist_ok=True)

        caminho_recorte = pasta_recorte_classe / nome_copiado
        caminho_anotada = pasta_conf_classe / nome_copiado

        recorte = imagem[y1:y2, x1:x2].copy()
        anotada = imagem.copy()
        cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 255, 0), 10)
        cv2.putText(
            anotada,
            classe,
            (max(10, x1), max(40, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

        if not salvar_imagem(caminho_recorte, recorte):
            erros.append((nome_copiado, "falha_ao_salvar_recorte"))
            continue

        if not salvar_imagem(caminho_anotada, anotada):
            erros.append((nome_copiado, "falha_ao_salvar_anotada"))
            continue

        df.loc[indice, "largura"] = largura
        df.loc[indice, "altura"] = altura
        df.loc[indice, "x1"] = x1
        df.loc[indice, "y1"] = y1
        df.loc[indice, "x2"] = x2
        df.loc[indice, "y2"] = y2
        df.loc[indice, "metodo"] = "ajuste_manual"
        df.loc[indice, "status_caixa"] = "ok"
        df.loc[indice, "erro"] = ""
        df.loc[indice, "arquivo_recortado"] = caminho_relativo(caminho_recorte)
        df.loc[indice, "arquivo_anotado"] = caminho_relativo(caminho_anotada)

        quantidade_aplicada += 1

    df.to_csv(CAMINHO_CAIXAS, index=False, encoding="utf-8-sig")

    print(f"Ajustes aplicados: {quantidade_aplicada}")

    if erros:
        print()
        print("Erros:")
        for nome, erro in erros:
            print(f"- {nome}: {erro}")

    print()
    print("Arquivo atualizado:")
    print(f"- {CAMINHO_CAIXAS}")
    print()
    print("Proximo passo:")
    print("python scripts\\13_conferir_caixas_automaticas.py")


if __name__ == "__main__":
    main()



