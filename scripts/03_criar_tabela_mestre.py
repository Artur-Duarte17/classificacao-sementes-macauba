from pathlib import Path
import pandas as pd


# ============================================================
# SCRIPT 03 - CRIAR TABELA MESTRE
# ------------------------------------------------------------
# Objetivo:
# - Juntar o inventário das imagens com os rótulos das planilhas
# - Usar como chave:
#     experimento + pasta/tratamento + ID da semente
# - Gerar:
#     tabela_mestre.csv
#     tabela_mestre_treinavel.csv
#     imagens_sem_rotulo.csv
#     rotulos_sem_imagem.csv
#
# Este script ainda NÃO copia imagens e NÃO treina IA.
# Ele apenas cruza imagem + rótulo.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"


def limpar_id(valor) -> str:
    """
    Padroniza IDs para comparação.
    Exemplos:
    - 'A1 ' -> 'a1'
    - 1.0 -> '1'
    - '001' -> '001'
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()

    if texto.endswith(".0"):
        texto = texto[:-2]

    return texto


def criar_chave(df: pd.DataFrame, col_experimento: str, col_pasta: str, col_id: str) -> pd.Series:
    """
    Cria uma chave textual única para fazer o cruzamento.
    """
    return (
        df[col_experimento].astype(str).str.strip()
        + "||"
        + df[col_pasta].astype(str).str.strip()
        + "||"
        + df[col_id].astype(str).str.strip().str.lower()
    )


def main():
    print("=" * 60)
    print("CRIANDO TABELA MESTRE")
    print("=" * 60)

    caminho_inventario = PASTA_TABELAS / "inventario_imagens.csv"
    caminho_rotulos = PASTA_TABELAS / "rotulos_planilhas.csv"

    if not caminho_inventario.exists():
        print("ERRO: inventario_imagens.csv não encontrado.")
        print(caminho_inventario)
        return

    if not caminho_rotulos.exists():
        print("ERRO: rotulos_planilhas.csv não encontrado.")
        print(caminho_rotulos)
        return

    inventario = pd.read_csv(caminho_inventario)
    rotulos = pd.read_csv(caminho_rotulos)

    print(f"Imagens no inventário: {len(inventario)}")
    print(f"Rótulos nas planilhas: {len(rotulos)}")
    print()

    # ------------------------------------------------------------
    # Prepara chaves do inventário
    # ------------------------------------------------------------
    inventario["id_busca_img"] = inventario["nome_sem_extensao"].apply(limpar_id)
    inventario["pasta_busca_img"] = inventario["pasta_pai"].astype(str).str.strip()
    inventario["experimento_busca_img"] = inventario["experimento"].astype(str).str.strip()

    inventario["chave"] = criar_chave(
        inventario,
        "experimento_busca_img",
        "pasta_busca_img",
        "id_busca_img"
    )

    # ------------------------------------------------------------
    # Prepara chaves dos rótulos
    # ------------------------------------------------------------
    rotulos["id_busca_rotulo"] = rotulos["id_busca"].apply(limpar_id)
    rotulos["pasta_busca_rotulo"] = rotulos["pasta_esperada"].astype(str).str.strip()
    rotulos["experimento_busca_rotulo"] = rotulos["experimento"].astype(str).str.strip()

    rotulos["chave"] = criar_chave(
        rotulos,
        "experimento_busca_rotulo",
        "pasta_busca_rotulo",
        "id_busca_rotulo"
    )

    # ------------------------------------------------------------
    # Verifica duplicatas de chave no inventário
    # Isso indicaria duas imagens competindo pelo mesmo rótulo.
    # ------------------------------------------------------------
    duplicatas_inventario = (
        inventario.groupby("chave")
        .size()
        .reset_index(name="quantidade")
    )

    duplicatas_inventario = duplicatas_inventario[
        duplicatas_inventario["quantidade"] > 1
    ].copy()

    caminho_dup_inv = PASTA_TABELAS / "duplicatas_inventario_chave.csv"
    duplicatas_inventario.to_csv(caminho_dup_inv, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Junta rótulos com imagens.
    # Usamos rótulos como base, porque só imagem com rótulo serve
    # para treinar.
    # ------------------------------------------------------------
    tabela = rotulos.merge(
        inventario,
        on="chave",
        how="left",
        suffixes=("_rotulo", "_img")
    )

    # Se não encontrou caminho_relativo, há rótulo sem imagem
    tabela["status"] = tabela["caminho_relativo"].apply(
        lambda x: "ok" if pd.notna(x) and str(x).strip() != "" else "sem_imagem"
    )

    # ------------------------------------------------------------
    # Imagens que existem, mas não têm rótulo nas planilhas
    # ------------------------------------------------------------
    imagens_sem_rotulo = inventario.merge(
        rotulos[["chave"]],
        on="chave",
        how="left",
        indicator=True
    )

    imagens_sem_rotulo = imagens_sem_rotulo[
        imagens_sem_rotulo["_merge"] == "left_only"
    ].copy()

    imagens_sem_rotulo["status"] = "sem_rotulo"

    caminho_imagens_sem_rotulo = PASTA_TABELAS / "imagens_sem_rotulo.csv"
    imagens_sem_rotulo.to_csv(
        caminho_imagens_sem_rotulo,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Rótulos sem imagem
    # ------------------------------------------------------------
    rotulos_sem_imagem = tabela[tabela["status"] == "sem_imagem"].copy()

    caminho_rotulos_sem_imagem = PASTA_TABELAS / "rotulos_sem_imagem.csv"
    rotulos_sem_imagem.to_csv(
        caminho_rotulos_sem_imagem,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Tabela mestre completa
    # ------------------------------------------------------------
    colunas_mestre = [
        "status",
        "classe",
        "contaminou",
        "germinou",
        "experimento_rotulo",
        "tratamento_planilha",
        "pasta_esperada",
        "id_semente_original",
        "id_busca",
        "caminho_relativo",
        "caminho_absoluto",
        "experimento_img",
        "pasta_pai",
        "nome_arquivo",
        "nome_sem_extensao",
        "extensao",
        "imagem_valida",
        "largura",
        "altura",
        "modo_cor",
        "origem_planilha",
        "qtd_observacoes",
        "observacao",
        "chave",
    ]

    # Algumas colunas podem mudar de nome dependendo do merge.
    # Ajuste seguro:
    renomear = {
        "experimento": "experimento_rotulo"
    }

    tabela = tabela.rename(columns=renomear)

    # Se o merge não gerou experimento_img automaticamente, cria a partir do inventário
    if "experimento_img" not in tabela.columns and "experimento" in tabela.columns:
        tabela["experimento_img"] = tabela["experimento"]

    # Garante que todas as colunas existam
    for coluna in colunas_mestre:
        if coluna not in tabela.columns:
            tabela[coluna] = ""

    tabela_mestre = tabela[colunas_mestre].copy()

    caminho_tabela_mestre = PASTA_TABELAS / "tabela_mestre.csv"
    tabela_mestre.to_csv(
        caminho_tabela_mestre,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Tabela treinável: apenas imagens encontradas e válidas
    # ------------------------------------------------------------
    tabela_treinavel = tabela_mestre[
        (tabela_mestre["status"] == "ok")
        & (tabela_mestre["imagem_valida"] == True)
        & (tabela_mestre["classe"].isin(["contaminada", "nao_contaminada"]))
    ].copy()

    caminho_treinavel = PASTA_TABELAS / "tabela_mestre_treinavel.csv"
    tabela_treinavel.to_csv(
        caminho_treinavel,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Resumos
    # ------------------------------------------------------------
    resumo_status = (
        tabela_mestre.groupby(["status", "classe"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["status", "classe"])
    )

    caminho_resumo_status = PASTA_TABELAS / "resumo_tabela_mestre.csv"
    resumo_status.to_csv(
        caminho_resumo_status,
        index=False,
        encoding="utf-8-sig"
    )

    resumo_treinavel = (
        tabela_treinavel.groupby(["experimento_rotulo", "pasta_esperada", "classe"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["experimento_rotulo", "pasta_esperada", "classe"])
    )

    caminho_resumo_treinavel = PASTA_TABELAS / "resumo_treinavel.csv"
    resumo_treinavel.to_csv(
        caminho_resumo_treinavel,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Resultado no terminal
    # ------------------------------------------------------------
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Rótulos totais: {len(rotulos)}")
    print(f"Imagens totais no inventário: {len(inventario)}")
    print(f"Registros na tabela mestre: {len(tabela_mestre)}")
    print(f"Registros treináveis: {len(tabela_treinavel)}")
    print(f"Rótulos sem imagem: {len(rotulos_sem_imagem)}")
    print(f"Imagens sem rótulo: {len(imagens_sem_rotulo)}")
    print(f"Duplicatas de chave no inventário: {len(duplicatas_inventario)}")

    print()
    print("Resumo da tabela mestre:")
    print(resumo_status.to_string(index=False))

    print()
    print("Resumo treinável:")
    if len(resumo_treinavel) > 0:
        print(resumo_treinavel.to_string(index=False))
    else:
        print("Nenhum registro treinável encontrado.")

    print()
    print("Arquivos gerados:")
    print(f"- {caminho_tabela_mestre}")
    print(f"- {caminho_treinavel}")
    print(f"- {caminho_imagens_sem_rotulo}")
    print(f"- {caminho_rotulos_sem_imagem}")
    print(f"- {caminho_resumo_status}")
    print(f"- {caminho_resumo_treinavel}")
    print(f"- {caminho_dup_inv}")

    print()
    print("Tabela mestre concluída.")


if __name__ == "__main__":
    main()