from pathlib import Path
import pandas as pd


# ============================================================
# SCRIPT 03 - CRIAR TABELA MESTRE
# ------------------------------------------------------------
# Objetivo:
# - Juntar o inventÃ¡rio das imagens com os rÃ³tulos das planilhas
# - Usar como chave:
#     experimento + pasta/tratamento + ID da semente
# - Gerar:
#     tabela_mestre.csv
#     tabela_mestre_treinavel.csv
#     imagens_sem_rotulo.csv
#     rotulos_sem_imagem.csv
#
# Este script ainda NÃƒO copia imagens e NÃƒO treina IA.
# Ele apenas cruza imagem + rÃ³tulo.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_INVENTARIO = PASTA_TABELAS / "01_inventario"
PASTA_ROTULOS = PASTA_TABELAS / "02_planilhas_rotulos"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"


def limpar_id(valor) -> str:
    """
    Padroniza IDs para comparaÃ§Ã£o.
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
    Cria uma chave textual Ãºnica para fazer o cruzamento.
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

    PASTA_TABELA_MESTRE.mkdir(parents=True, exist_ok=True)

    caminho_inventario = PASTA_INVENTARIO / "inventario_imagens.csv"
    caminho_rotulos = PASTA_ROTULOS / "rotulos_planilhas.csv"

    if not caminho_inventario.exists():
        print("ERRO: inventario_imagens.csv nÃ£o encontrado.")
        print(caminho_inventario)
        return

    if not caminho_rotulos.exists():
        print("ERRO: rotulos_planilhas.csv nÃ£o encontrado.")
        print(caminho_rotulos)
        return

    inventario = pd.read_csv(caminho_inventario)
    rotulos = pd.read_csv(caminho_rotulos)

    print(f"Imagens no inventÃ¡rio: {len(inventario)}")
    print(f"RÃ³tulos nas planilhas: {len(rotulos)}")
    print()

    # ------------------------------------------------------------
    # Prepara chaves do inventÃ¡rio
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
    # Prepara chaves dos rÃ³tulos
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
    # Verifica duplicatas de chave no inventÃ¡rio
    # Isso indicaria duas imagens competindo pelo mesmo rÃ³tulo.
    # ------------------------------------------------------------
    duplicatas_inventario = (
        inventario.groupby("chave")
        .size()
        .reset_index(name="quantidade")
    )

    duplicatas_inventario = duplicatas_inventario[
        duplicatas_inventario["quantidade"] > 1
    ].copy()

    caminho_dup_inv = PASTA_TABELA_MESTRE / "duplicatas_inventario_chave.csv"
    duplicatas_inventario.to_csv(caminho_dup_inv, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Junta rÃ³tulos com imagens.
    # Usamos rÃ³tulos como base, porque sÃ³ imagem com rÃ³tulo serve
    # para treinar.
    # ------------------------------------------------------------
    tabela = rotulos.merge(
        inventario,
        on="chave",
        how="left",
        suffixes=("_rotulo", "_img")
    )

    # Se nÃ£o encontrou caminho_relativo, hÃ¡ rÃ³tulo sem imagem
    tabela["status"] = tabela["caminho_relativo"].apply(
        lambda x: "ok" if pd.notna(x) and str(x).strip() != "" else "sem_imagem"
    )

    # ------------------------------------------------------------
    # Imagens que existem, mas nÃ£o tÃªm rÃ³tulo nas planilhas
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

    caminho_imagens_sem_rotulo = PASTA_TABELA_MESTRE / "imagens_sem_rotulo.csv"
    imagens_sem_rotulo.to_csv(
        caminho_imagens_sem_rotulo,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # RÃ³tulos sem imagem
    # ------------------------------------------------------------
    rotulos_sem_imagem = tabela[tabela["status"] == "sem_imagem"].copy()

    caminho_rotulos_sem_imagem = PASTA_TABELA_MESTRE / "rotulos_sem_imagem.csv"
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

    # Se o merge nÃ£o gerou experimento_img automaticamente, cria a partir do inventÃ¡rio
    if "experimento_img" not in tabela.columns and "experimento" in tabela.columns:
        tabela["experimento_img"] = tabela["experimento"]

    # Garante que todas as colunas existam
    for coluna in colunas_mestre:
        if coluna not in tabela.columns:
            tabela[coluna] = ""

    tabela_mestre = tabela[colunas_mestre].copy()

    caminho_tabela_mestre = PASTA_TABELA_MESTRE / "tabela_mestre.csv"
    tabela_mestre.to_csv(
        caminho_tabela_mestre,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Tabela treinÃ¡vel: apenas imagens encontradas e vÃ¡lidas
    # ------------------------------------------------------------
    tabela_treinavel = tabela_mestre[
        (tabela_mestre["status"] == "ok")
        & (tabela_mestre["imagem_valida"] == True)
        & (tabela_mestre["classe"].isin(["contaminada", "nao_contaminada"]))
    ].copy()

    caminho_treinavel = PASTA_TABELA_MESTRE / "tabela_mestre_treinavel.csv"
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

    caminho_resumo_status = PASTA_TABELA_MESTRE / "resumo_tabela_mestre.csv"
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

    caminho_resumo_treinavel = PASTA_TABELA_MESTRE / "resumo_treinavel.csv"
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
    print(f"RÃ³tulos totais: {len(rotulos)}")
    print(f"Imagens totais no inventÃ¡rio: {len(inventario)}")
    print(f"Registros na tabela mestre: {len(tabela_mestre)}")
    print(f"Registros treinÃ¡veis: {len(tabela_treinavel)}")
    print(f"RÃ³tulos sem imagem: {len(rotulos_sem_imagem)}")
    print(f"Imagens sem rÃ³tulo: {len(imagens_sem_rotulo)}")
    print(f"Duplicatas de chave no inventÃ¡rio: {len(duplicatas_inventario)}")

    print()
    print("Resumo da tabela mestre:")
    print(resumo_status.to_string(index=False))

    print()
    print("Resumo treinÃ¡vel:")
    if len(resumo_treinavel) > 0:
        print(resumo_treinavel.to_string(index=False))
    else:
        print("Nenhum registro treinÃ¡vel encontrado.")

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
    print("Tabela mestre concluÃ­da.")


if __name__ == "__main__":
    main()


