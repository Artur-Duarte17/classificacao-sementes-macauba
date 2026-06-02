from pathlib import Path

import pandas as pd


# ============================================================
# SCRIPT 10 - JUNTAR RELATORIOS DE CAIXAS AUTOMATICAS
# ------------------------------------------------------------
# Objetivo:
# - Juntar as caixas de Micro-ondas e Piloto/TESTE 2
# - Gerar o arquivo unico usado pelos scripts 09 e 10
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"

ARQUIVOS_ENTRADA = [
    ("microondas", PASTA_TABELAS / "caixas_microondas.csv"),
    ("piloto_teste2", PASTA_TABELAS / "caixas_piloto_teste2.csv"),
]

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


def main():
    print("=" * 60)
    print("JUNTANDO CAIXAS AUTOMATICAS")
    print("=" * 60)

    tabelas = []

    for origem, caminho in ARQUIVOS_ENTRADA:
        if not caminho.exists():
            print(f"ERRO: arquivo nao encontrado: {caminho}")
            print("Execute primeiro:")
            print("python scripts\\08_gerar_caixas_microondas.py")
            print("python scripts\\09_gerar_caixas_piloto_teste2.py")
            return

        df = pd.read_csv(caminho)

        if "origem_detector" not in df.columns:
            df["origem_detector"] = origem

        tabelas.append(df)

    combinado = pd.concat(tabelas, ignore_index=True)

    for coluna in COLUNAS_RELATORIO:
        if coluna not in combinado.columns:
            combinado[coluna] = ""

    combinado = combinado[COLUNAS_RELATORIO].copy()

    caminho_saida = PASTA_TABELAS / "caixas_automaticas.csv"
    combinado.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    if len(combinado):
        resumo = (
            combinado.groupby(["origem_detector", "classe", "status_caixa"])
            .size()
            .reset_index(name="quantidade")
            .sort_values(["origem_detector", "classe", "status_caixa"])
        )
        print(resumo.to_string(index=False))
    else:
        print("Nenhuma caixa encontrada nos relatorios de entrada.")

    print()
    print("Arquivo gerado:")
    print(f"- {caminho_saida}")
    print()


if __name__ == "__main__":
    main()
