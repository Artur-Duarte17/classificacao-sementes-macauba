from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm


# ============================================================
# SCRIPT 04 - CRIAR DATASET BINÃRIO
# ------------------------------------------------------------
# Objetivo:
# - Ler tabela_mestre_treinavel.csv
# - Copiar as imagens para:
#     saidas/dataset_binario/contaminada
#     saidas/dataset_binario/nao_contaminada
# - Criar nomes seguros preservando a origem
# - Gerar relatÃ³rio de cÃ³pia
#
# Este script NÃƒO altera as imagens originais.
# Ele apenas copia imagens para uma nova pasta organizada.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_binario"


def nome_seguro(caminho_relativo: str) -> str:
    """
    Transforma o caminho relativo em um nome de arquivo seguro.

    Exemplo:
    TESTE 2/T3/a1.jpg
    vira:
    TESTE_2__T3__a1.jpg
    """
    texto = str(caminho_relativo)

    substituicoes = {
        "\\": "__",
        "/": "__",
        " ": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }

    for antigo, novo in substituicoes.items():
        texto = texto.replace(antigo, novo)

    return texto


def main():
    print("=" * 60)
    print("CRIANDO DATASET BINÃRIO")
    print("=" * 60)

    PASTA_DATASET_TABELAS.mkdir(parents=True, exist_ok=True)

    caminho_tabela = PASTA_TABELA_MESTRE / "tabela_mestre_treinavel.csv"

    if not caminho_tabela.exists():
        print("ERRO: tabela_mestre_treinavel.csv nÃ£o encontrada.")
        print(caminho_tabela)
        return

    df = pd.read_csv(caminho_tabela)

    print(f"Registros na tabela treinÃ¡vel: {len(df)}")

    # MantÃ©m apenas classes esperadas
    df = df[df["classe"].isin(["contaminada", "nao_contaminada"])].copy()

    print(f"Registros apÃ³s filtro de classe: {len(df)}")
    print()

    # Cria pastas de saÃ­da
    pasta_contaminada = PASTA_DATASET / "contaminada"
    pasta_nao_contaminada = PASTA_DATASET / "nao_contaminada"

    pasta_contaminada.mkdir(parents=True, exist_ok=True)
    pasta_nao_contaminada.mkdir(parents=True, exist_ok=True)

    registros = []

    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Copiando imagens"):
        classe = linha["classe"]
        origem = Path(str(linha["caminho_absoluto"]))
        caminho_relativo = str(linha["caminho_relativo"])

        nome_destino = nome_seguro(caminho_relativo)

        if classe == "contaminada":
            destino = pasta_contaminada / nome_destino
        else:
            destino = pasta_nao_contaminada / nome_destino

        status_copia = "ok"
        erro = ""

        try:
            if not origem.exists():
                status_copia = "erro"
                erro = "arquivo_origem_nao_existe"
            else:
                shutil.copy2(origem, destino)

        except Exception as e:
            status_copia = "erro"
            erro = str(e)

        registros.append({
            "classe": classe,
            "caminho_original": str(origem),
            "caminho_relativo_original": caminho_relativo,
            "arquivo_copiado": str(destino),
            "nome_copiado": nome_destino,
            "status_copia": status_copia,
            "erro": erro,
        })

    relatorio = pd.DataFrame(registros)

    caminho_relatorio = PASTA_DATASET_TABELAS / "relatorio_copia_dataset_binario.csv"
    relatorio.to_csv(caminho_relatorio, index=False, encoding="utf-8-sig")

    resumo = (
        relatorio.groupby(["classe", "status_copia"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["classe", "status_copia"])
    )

    caminho_resumo = PASTA_DATASET_TABELAS / "resumo_copia_dataset_binario.csv"
    resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(resumo.to_string(index=False))

    print()
    print("Pastas criadas:")
    print(f"- {pasta_contaminada}")
    print(f"- {pasta_nao_contaminada}")

    print()
    print("Arquivos gerados:")
    print(f"- {caminho_relatorio}")
    print(f"- {caminho_resumo}")

    print()
    print("Dataset binÃ¡rio concluÃ­do.")


if __name__ == "__main__":
    main()


