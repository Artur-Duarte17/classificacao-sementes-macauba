from pathlib import Path
import pandas as pd


# ============================================================
# SCRIPT 01 - INSPECIONAR PLANILHAS
# ------------------------------------------------------------
# Objetivo:
# - Ler todas as planilhas da pasta dados_originais/planilhas
# - Identificar abas, colunas e quantidade de linhas
# - Gerar arquivos CSV de conferência
#
# Este script ainda NÃO cria os rótulos finais.
# Ele apenas mostra como as planilhas estão estruturadas.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_PLANILHAS = PASTA_PROJETO / "dados_originais" / "planilhas"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "tabelas"

EXTENSOES_PLANILHA = {".xlsx", ".xls"}


def limpar_nome_arquivo(texto: str) -> str:
    """
    Remove caracteres problemáticos para usar em nomes de arquivos.
    """
    texto = str(texto)
    caracteres_invalidos = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for c in caracteres_invalidos:
        texto = texto.replace(c, "_")
    return texto.strip()


def main():
    print("=" * 60)
    print("INSPEÇÃO DAS PLANILHAS")
    print("=" * 60)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    if not PASTA_PLANILHAS.exists():
        print("ERRO: pasta de planilhas não encontrada:")
        print(PASTA_PLANILHAS)
        return

    planilhas = [
        caminho
        for caminho in PASTA_PLANILHAS.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_PLANILHA
    ]

    print(f"Pasta de planilhas: {PASTA_PLANILHAS}")
    print(f"Total de planilhas encontradas: {len(planilhas)}")
    print()

    registros_resumo = []
    registros_colunas = []

    for caminho_planilha in planilhas:
        print("-" * 60)
        print(f"Lendo: {caminho_planilha.name}")

        try:
            excel = pd.ExcelFile(caminho_planilha)
            abas = excel.sheet_names

        except Exception as e:
            print(f"ERRO ao abrir planilha: {e}")

            registros_resumo.append({
                "arquivo": caminho_planilha.name,
                "aba": "",
                "linhas": None,
                "colunas": None,
                "erro": str(e)
            })
            continue

        print(f"Abas encontradas: {abas}")

        for aba in abas:
            try:
                df = pd.read_excel(caminho_planilha, sheet_name=aba)

                # Remove linhas e colunas totalmente vazias
                df = df.dropna(how="all")
                df = df.dropna(axis=1, how="all")

                qtd_linhas = len(df)
                qtd_colunas = len(df.columns)

                print(f"  Aba: {aba} | Linhas: {qtd_linhas} | Colunas: {qtd_colunas}")

                registros_resumo.append({
                    "arquivo": caminho_planilha.name,
                    "aba": aba,
                    "linhas": qtd_linhas,
                    "colunas": qtd_colunas,
                    "erro": ""
                })

                for posicao, coluna in enumerate(df.columns, start=1):
                    registros_colunas.append({
                        "arquivo": caminho_planilha.name,
                        "aba": aba,
                        "posicao_coluna": posicao,
                        "nome_coluna": str(coluna),
                        "tipo_detectado": str(df[coluna].dtype),
                        "valores_nao_vazios": int(df[coluna].notna().sum()),
                        "valores_vazios": int(df[coluna].isna().sum())
                    })

                # Salva uma prévia das primeiras linhas de cada aba
                nome_base = limpar_nome_arquivo(caminho_planilha.stem)
                nome_aba = limpar_nome_arquivo(aba)

                caminho_previa = PASTA_SAIDA / f"previa__{nome_base}__{nome_aba}.csv"
                df.head(20).to_csv(caminho_previa, index=False, encoding="utf-8-sig")

            except Exception as e:
                print(f"  ERRO ao ler aba {aba}: {e}")

                registros_resumo.append({
                    "arquivo": caminho_planilha.name,
                    "aba": aba,
                    "linhas": None,
                    "colunas": None,
                    "erro": str(e)
                })

    df_resumo = pd.DataFrame(registros_resumo)
    df_colunas = pd.DataFrame(registros_colunas)

    caminho_resumo = PASTA_SAIDA / "resumo_planilhas.csv"
    caminho_colunas = PASTA_SAIDA / "colunas_planilhas.csv"

    df_resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")
    df_colunas.to_csv(caminho_colunas, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("ARQUIVOS GERADOS")
    print("=" * 60)
    print(caminho_resumo)
    print(caminho_colunas)
    print()
    print("Também foram gerados arquivos 'previa__...' com as primeiras linhas de cada aba.")
    print("Inspeção concluída.")


if __name__ == "__main__":
    main()