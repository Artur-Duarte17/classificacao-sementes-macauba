from pathlib import Path

import pandas as pd


# ============================================================
# SCRIPT 30 - CRIAR TABELA INTEGRADA
# ------------------------------------------------------------
# Objetivo:
# - Consolidar metadados, split e predicoes em uma tabela unica
# - Juntar tabela mestre, split e predicoes dos modelos ja treinados
# - Criar uma primeira coluna de triagem: alto_risco/baixo_risco/incerto
#
# Este script nao treina modelo e nao altera imagens.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_BASELINE_TABELAS = PASTA_TABELAS / "06_modelos" / "baseline"
PASTA_RECORTES_TABELAS = PASTA_TABELAS / "06_modelos" / "recortes"
PASTA_TRIAGEM_TABELAS = PASTA_TABELAS / "08_triagem"

LIMIAR_ALTO_RISCO = 0.70
LIMIAR_BAIXO_RISCO = 0.30


def nome_seguro(caminho_relativo: str) -> str:
    """Replica a regra usada no script 04 para nomear imagens copiadas."""
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


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho)


def ler_csv_opcional(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        print(f"AVISO: arquivo opcional nao encontrado: {caminho}")
        return pd.DataFrame()
    return pd.read_csv(caminho)


def preparar_predicoes(
    df: pd.DataFrame,
    prefixo: str,
    coluna_prob: str = "prob_contaminada",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["nome_copiado", f"prob_{prefixo}"])

    df = df.copy()
    df["nome_copiado"] = df["caminho_imagem"].map(lambda x: Path(str(x)).name)

    colunas = ["nome_copiado"]
    renomear = {}

    if coluna_prob in df.columns:
        colunas.append(coluna_prob)
        renomear[coluna_prob] = f"prob_{prefixo}"

    for coluna in [
        "classe_real",
        "alvo",
        "predito_threshold_0_50",
        "predito_threshold_melhor_f1_validacao",
        "predito_threshold_prioridade_recall_validacao",
    ]:
        if coluna in df.columns:
            colunas.append(coluna)
            renomear[coluna] = f"{coluna}_{prefixo}"

    return df[colunas].rename(columns=renomear)


def classificar_triagem(probabilidade):
    if pd.isna(probabilidade):
        return "sem_predicao"
    if probabilidade >= LIMIAR_ALTO_RISCO:
        return "alto_risco"
    if probabilidade <= LIMIAR_BAIXO_RISCO:
        return "baixo_risco"
    return "incerto"


def main():
    print("=" * 60)
    print("CRIANDO TABELA INTEGRADA")
    print("=" * 60)

    PASTA_TRIAGEM_TABELAS.mkdir(parents=True, exist_ok=True)

    tabela = ler_csv_obrigatorio(PASTA_TABELA_MESTRE / "tabela_mestre.csv")
    print(f"Registros na tabela mestre: {len(tabela)}")

    tabela = tabela.copy()
    tabela["nome_copiado"] = tabela["caminho_relativo"].map(nome_seguro)

    relatorio_copia = ler_csv_opcional(
        PASTA_DATASET_TABELAS / "relatorio_copia_dataset_binario.csv"
    )
    if not relatorio_copia.empty:
        relatorio_copia = relatorio_copia[
            ["caminho_relativo_original", "nome_copiado", "status_copia"]
        ].drop_duplicates()
        tabela = tabela.merge(
            relatorio_copia,
            left_on="caminho_relativo",
            right_on="caminho_relativo_original",
            how="left",
            suffixes=("", "_relatorio"),
        )
        tabela["nome_copiado"] = tabela["nome_copiado_relatorio"].fillna(
            tabela["nome_copiado"]
        )
        tabela = tabela.drop(
            columns=["caminho_relativo_original", "nome_copiado_relatorio"],
            errors="ignore",
        )
    else:
        tabela["status_copia"] = pd.NA

    split = ler_csv_opcional(PASTA_DATASET_TABELAS / "divisao_treino_validacao_teste.csv")
    if not split.empty:
        split = split[["nome_arquivo", "split"]].drop_duplicates("nome_arquivo")
        mapa_split = split.set_index("nome_arquivo")["split"]
        tabela["split"] = tabela["nome_copiado"].map(mapa_split)
    else:
        tabela["split"] = pd.NA

    pred_baseline = preparar_predicoes(
        ler_csv_opcional(PASTA_BASELINE_TABELAS / "predicoes_baseline_resnet18_teste.csv"),
        "baseline_resnet18",
    )
    pred_recortes = preparar_predicoes(
        ler_csv_opcional(PASTA_RECORTES_TABELAS / "predicoes_recortes_resnet18_teste.csv"),
        "recortes_resnet18",
    )

    tabela = tabela.merge(pred_baseline, on="nome_copiado", how="left")
    tabela = tabela.merge(pred_recortes, on="nome_copiado", how="left")

    colunas_prob = [
        coluna
        for coluna in ["prob_baseline_resnet18", "prob_recortes_resnet18"]
        if coluna in tabela.columns
    ]
    tabela["prob_media_modelos"] = tabela[colunas_prob].mean(axis=1, skipna=True)
    tabela["triagem_preliminar"] = tabela["prob_media_modelos"].map(classificar_triagem)

    colunas_inicio = [
        "status",
        "classe",
        "contaminou",
        "germinou",
        "experimento_rotulo",
        "tratamento_planilha",
        "pasta_esperada",
        "id_semente_original",
        "caminho_relativo",
        "caminho_absoluto",
        "nome_copiado",
        "split",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "prob_media_modelos",
        "triagem_preliminar",
    ]
    colunas_inicio = [coluna for coluna in colunas_inicio if coluna in tabela.columns]
    colunas_restantes = [coluna for coluna in tabela.columns if coluna not in colunas_inicio]
    tabela = tabela[colunas_inicio + colunas_restantes]

    caminho_saida = PASTA_TRIAGEM_TABELAS / "tabela_integrada.csv"
    tabela.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    resumo_triagem = (
        tabela.groupby(["split", "triagem_preliminar"], dropna=False)
        .size()
        .reset_index(name="quantidade")
        .sort_values(["split", "triagem_preliminar"])
    )
    caminho_resumo = PASTA_TRIAGEM_TABELAS / "resumo_triagem_preliminar.csv"
    resumo_triagem.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")

    print()
    print("Resumo da triagem preliminar:")
    print(resumo_triagem.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {caminho_saida}")
    print(f"- {caminho_resumo}")
    print()
    print("Tabela integrada concluida.")


if __name__ == "__main__":
    main()



