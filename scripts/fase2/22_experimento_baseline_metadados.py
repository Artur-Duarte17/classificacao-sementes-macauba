# -*- coding: utf-8 -*-
r"""
22_experimento_baseline_metadados.py

Objetivo:
    Testar se metadados da tabela mestre (origem, tratamento, pasta e campos
    derivados) ja explicam boa parte da predicao de contaminacao.

Uso recomendado no projeto:
    python scripts\fase2\22_experimento_baseline_metadados.py

Uso com caminhos customizados:
    python scripts\fase2\22_experimento_baseline_metadados.py ^
        --tabelas-dir saidas\tabelas ^
        --saida-dir saidas\tabelas\08_baseline_metadados

Saidas geradas:
    saidas\tabelas\08_baseline_metadados\comparacao_baseline_metadados_modelos.csv
    saidas\tabelas\08_baseline_metadados\associacao_metadados_contaminacao.csv
    saidas\tabelas\08_baseline_metadados\taxas_contaminacao_por_grupo.csv
    saidas\tabelas\08_baseline_metadados\predicoes_baseline_metadados_teste.csv
    saidas\tabelas\08_baseline_metadados\conclusao_baseline_metadados.txt

Interpretacao:
    Se um modelo que usa somente metadados chegar perto ou superar os modelos
    de imagem, isso indica que lote/tratamento/origem/pasta podem estar
    explicando parte relevante da predicao. Isso nao prova erro no projeto,
    mas alerta para vies de lote/tratamento.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSE_POSITIVA = "contaminada"


def ler_csv(caminho: Path) -> pd.DataFrame:
    """Le um CSV removendo BOM e espacos dos nomes das colunas."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def salvar_csv(df: pd.DataFrame, caminho: Path) -> None:
    """Salva CSV com utf-8-sig para abrir melhor no Excel/Windows."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")


def metricas_binarias(y_true: Iterable[int], prob: Iterable[float], threshold: float) -> Dict[str, float]:
    """
    Calcula metricas considerando a classe contaminada como positiva.

    y_true:
        1 = contaminada
        0 = nao_contaminada
    prob:
        probabilidade/score estimado para contaminada.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) else 0.0
    acuracia = (tp + tn) / (tp + tn + fp + fn)

    return {
        "threshold": float(threshold),
        "acuracia": float(acuracia),
        "precisao_contaminada": float(precisao),
        "recall_contaminada": float(recall),
        "sensibilidade_contaminada": float(recall),
        "especificidade_nao_contaminada": float(especificidade),
        "f1_contaminada": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def escolher_threshold(
    y_validacao: Iterable[int],
    prob_validacao: Iterable[float],
    modo: str,
) -> float:
    """
    Escolhe o threshold usando apenas a validacao.

    modo = "melhor_f1":
        maximiza F1 da classe contaminada.

    modo = "prioridade_recall":
        tenta manter recall >= 0.95 e, dentro disso, maximiza F1 e especificidade.
        Se nao houver threshold com recall >= 0.95, escolhe o maior recall possivel.
    """
    y_validacao = np.asarray(y_validacao).astype(int)
    prob_validacao = np.asarray(prob_validacao).astype(float)

    thresholds = np.unique(np.round(np.r_[np.linspace(0, 1, 201), prob_validacao], 6))
    linhas = [metricas_binarias(y_validacao, prob_validacao, t) for t in thresholds]
    tabela = pd.DataFrame(linhas)

    if modo == "melhor_f1":
        ordenada = tabela.sort_values(
            ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada", "threshold"],
            ascending=[False, False, False, False],
        )
        return float(ordenada.iloc[0]["threshold"])

    if modo == "prioridade_recall":
        candidatas = tabela[tabela["recall_contaminada"] >= 0.95]
        if candidatas.empty:
            ordenada = tabela.sort_values(
                ["recall_contaminada", "f1_contaminada", "threshold"],
                ascending=[False, False, False],
            )
        else:
            ordenada = candidatas.sort_values(
                ["f1_contaminada", "especificidade_nao_contaminada", "threshold"],
                ascending=[False, False, False],
            )
        return float(ordenada.iloc[0]["threshold"])

    raise ValueError(f"Modo de threshold desconhecido: {modo}")


def extrair_letra_id(valor: object) -> str:
    """Extrai a parte alfabetica inicial de IDs como a1, b4, j10."""
    if pd.isna(valor):
        return "desconhecido"

    texto = str(valor).strip()
    achou = re.match(r"([A-Za-z]+)", texto)
    if not achou:
        return "sem_letra"

    return achou.group(1).lower()


def extrair_numero_id(valor: object) -> float:
    """Extrai o ultimo numero de IDs como a1, b4, j10."""
    if pd.isna(valor):
        return np.nan

    numeros = re.findall(r"\d+", str(valor))
    if not numeros:
        return np.nan

    return float(numeros[-1])


def preparar_tabela_com_split(tabelas_dir: Path) -> pd.DataFrame:
    """
    Liga tabela_mestre_treinavel aos splits usados nos modelos de imagem.

    Usa:
        03_tabela_mestre/tabela_mestre_treinavel.csv
        04_dataset_split/relatorio_copia_dataset_binario.csv
        04_dataset_split/divisao_treino_validacao_teste.csv
    """
    tabela_mestre = ler_csv(tabelas_dir / "03_tabela_mestre" / "tabela_mestre_treinavel.csv")
    relatorio_copia = ler_csv(tabelas_dir / "04_dataset_split" / "relatorio_copia_dataset_binario.csv")
    split = ler_csv(tabelas_dir / "04_dataset_split" / "divisao_treino_validacao_teste.csv")

    mapa_nome = relatorio_copia[["caminho_relativo_original", "nome_copiado"]].rename(
        columns={"caminho_relativo_original": "caminho_relativo"}
    )

    df = tabela_mestre.merge(mapa_nome, on="caminho_relativo", how="left", validate="one_to_one")

    split_reduzido = split[["nome_arquivo", "split", "alvo"]].rename(
        columns={"nome_arquivo": "nome_copiado", "alvo": "alvo_split"}
    )
    df = df.merge(split_reduzido, on="nome_copiado", how="left", validate="one_to_one")

    if df["split"].isna().any():
        faltando = int(df["split"].isna().sum())
        raise ValueError(f"{faltando} registros da tabela mestre ficaram sem split.")

    df["contaminou"] = pd.to_numeric(df["contaminou"], errors="coerce").astype(int)

    if (df["contaminou"] != df["alvo_split"]).any():
        divergencias = int((df["contaminou"] != df["alvo_split"]).sum())
        raise ValueError(f"Ha {divergencias} divergencias entre contaminou e alvo do split.")

    return df


def criar_features_metadados(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Cria features usando apenas metadados/campos derivados.
    Nao usa imagem, classe, contaminou, germinou, caminho absoluto nem dados de predicao.
    """
    dados = df.copy()

    dados["id_letra"] = dados["id_busca"].apply(extrair_letra_id)
    dados["id_numero"] = dados["id_busca"].apply(extrair_numero_id)

    dados["largura"] = pd.to_numeric(dados["largura"], errors="coerce")
    dados["altura"] = pd.to_numeric(dados["altura"], errors="coerce")
    dados["qtd_observacoes"] = pd.to_numeric(dados["qtd_observacoes"], errors="coerce")
    dados["area_mp"] = (dados["largura"] * dados["altura"]) / 1_000_000
    dados["aspect_ratio"] = dados["largura"] / dados["altura"]

    partes = dados["caminho_relativo"].astype(str).str.replace("\\", "/", regex=False).str.split("/", expand=True)
    dados["path_nivel_0"] = partes[0] if 0 in partes.columns else np.nan
    dados["path_nivel_1"] = partes[1] if 1 in partes.columns else np.nan

    colunas_categoricas = [
        "experimento_rotulo",
        "tratamento_planilha",
        "pasta_esperada",
        "experimento_img",
        "pasta_pai",
        "extensao",
        "modo_cor",
        "origem_planilha",
        "id_letra",
        "path_nivel_0",
        "path_nivel_1",
    ]

    colunas_numericas = [
        "largura",
        "altura",
        "area_mp",
        "aspect_ratio",
        "qtd_observacoes",
        "id_numero",
    ]

    existentes_cat = [c for c in colunas_categoricas if c in dados.columns]
    existentes_num = [c for c in colunas_numericas if c in dados.columns]

    return dados[existentes_cat + existentes_num], existentes_cat, existentes_num


def criar_preprocessador(colunas_categoricas: List[str], colunas_numericas: List[str]) -> ColumnTransformer:
    """Pre-processamento para modelos do sklearn."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                colunas_categoricas,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                colunas_numericas,
            ),
        ]
    )


def fit_prior_grupo(
    df_treino: pd.DataFrame,
    y_treino: pd.Series,
    colunas_grupo: List[str],
    suavizacao: float = 5.0,
) -> Tuple[Dict[str, float], float]:
    """
    Modelo simples: probabilidade de contaminacao = taxa historica do grupo.
    Usa suavizacao para evitar taxa 0% ou 100% em grupos pequenos.
    """
    taxa_global = float(y_treino.mean())

    temp = df_treino[colunas_grupo].fillna("desconhecido").astype(str).copy()
    temp["_y"] = y_treino.values
    temp["_chave_grupo"] = temp[colunas_grupo].agg("||".join, axis=1)

    stats = temp.groupby("_chave_grupo")["_y"].agg(["sum", "count"])
    stats["taxa_suavizada"] = (stats["sum"] + suavizacao * taxa_global) / (stats["count"] + suavizacao)

    return stats["taxa_suavizada"].to_dict(), taxa_global


def predict_prior_grupo(
    df_avaliacao: pd.DataFrame,
    mapa_taxa: Dict[str, float],
    taxa_global: float,
    colunas_grupo: List[str],
) -> np.ndarray:
    """Prediz usando a taxa historica do grupo; se grupo novo, usa taxa global."""
    temp = df_avaliacao[colunas_grupo].fillna("desconhecido").astype(str).copy()
    chaves = temp.agg("||".join, axis=1)
    return chaves.map(mapa_taxa).fillna(taxa_global).to_numpy(dtype=float)


def avaliar_modelo(
    nome_modelo: str,
    tipo: str,
    y_validacao: pd.Series,
    prob_validacao: np.ndarray,
    y_teste: pd.Series,
    prob_teste: np.ndarray,
) -> List[Dict[str, object]]:
    """Avalia threshold fixo, melhor F1 e prioridade de recall."""
    avaliacoes = []

    cenarios = [
        ("teste_threshold_0_50", 0.50),
        ("teste_threshold_melhor_f1_validacao", escolher_threshold(y_validacao, prob_validacao, "melhor_f1")),
        (
            "teste_threshold_prioridade_recall_validacao",
            escolher_threshold(y_validacao, prob_validacao, "prioridade_recall"),
        ),
    ]

    for cenario, threshold in cenarios:
        linha = metricas_binarias(y_teste, prob_teste, threshold)
        linha["modelo"] = nome_modelo
        linha["tipo"] = tipo
        linha["cenario"] = cenario
        avaliacoes.append(linha)

    return avaliacoes


def associacao_categorica(df: pd.DataFrame, colunas: List[str], alvo: str = "contaminou") -> pd.DataFrame:
    """
    Mede associacao entre metadados categoricos e contaminacao.

    Usa Cramer's V:
        perto de 0 = pouca associacao
        acima de 0.30 = associacao relevante para este tipo de alerta
        acima de 0.50 = associacao forte

    O p-valor do qui-quadrado e gerado se scipy estiver disponivel.
    """
    try:
        from scipy.stats import chi2_contingency
    except Exception:  # pragma: no cover
        chi2_contingency = None

    linhas = []
    y = df[alvo].astype(int)

    for coluna in colunas:
        if coluna not in df.columns:
            continue

        tab = pd.crosstab(df[coluna].fillna("desconhecido").astype(str), y)

        if tab.shape[0] < 2 or tab.shape[1] < 2:
            linhas.append(
                {
                    "campo": coluna,
                    "n_valores": int(df[coluna].nunique(dropna=False)),
                    "cramers_v": np.nan,
                    "p_chi2": np.nan,
                    "interpretacao": "sem variacao suficiente",
                }
            )
            continue

        if chi2_contingency is None:
            cramers_v = np.nan
            p_valor = np.nan
        else:
            chi2, p_valor, _, _ = chi2_contingency(tab)
            n = tab.to_numpy().sum()
            r, k = tab.shape
            phi2 = chi2 / n

            # Correcao de vies do Cramer's V.
            phi2_corrigido = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            r_corrigido = r - ((r - 1) ** 2) / (n - 1)
            k_corrigido = k - ((k - 1) ** 2) / (n - 1)
            denominador = min((k_corrigido - 1), (r_corrigido - 1))
            cramers_v = float(np.sqrt(phi2_corrigido / denominador)) if denominador > 0 else np.nan

        if pd.isna(cramers_v):
            interpretacao = "nao calculado"
        elif cramers_v >= 0.50:
            interpretacao = "associacao forte"
        elif cramers_v >= 0.30:
            interpretacao = "associacao relevante"
        elif cramers_v >= 0.10:
            interpretacao = "associacao fraca/moderada"
        else:
            interpretacao = "associacao baixa"

        linhas.append(
            {
                "campo": coluna,
                "n_valores": int(df[coluna].nunique(dropna=False)),
                "cramers_v": cramers_v,
                "p_chi2": p_valor,
                "interpretacao": interpretacao,
            }
        )

    return pd.DataFrame(linhas).sort_values("cramers_v", ascending=False)


def taxas_por_grupo(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """Gera taxas de contaminacao por origem/tratamento/pasta."""
    linhas = []

    for coluna in colunas:
        if coluna not in df.columns:
            continue

        resumo = (
            df.groupby(coluna, dropna=False)["contaminou"]
            .agg(total="count", contaminadas="sum")
            .reset_index()
            .rename(columns={coluna: "valor"})
        )
        resumo["campo"] = coluna
        resumo["nao_contaminadas"] = resumo["total"] - resumo["contaminadas"]
        resumo["taxa_contaminacao"] = resumo["contaminadas"] / resumo["total"]

        linhas.append(resumo[["campo", "valor", "total", "contaminadas", "nao_contaminadas", "taxa_contaminacao"]])

    if not linhas:
        return pd.DataFrame()

    return pd.concat(linhas, ignore_index=True).sort_values(
        ["campo", "taxa_contaminacao", "total"],
        ascending=[True, False, False],
    )


def carregar_comparacao_imagem(tabelas_dir: Path) -> pd.DataFrame:
    """Carrega a comparacao ja existente dos modelos de imagem, se existir."""
    caminho = tabelas_dir / "06_modelos" / "comparacao" / "comparacao_modelos_teste.csv"

    if not caminho.exists():
        return pd.DataFrame()

    df = ler_csv(caminho)

    if "modelo" not in df.columns:
        return pd.DataFrame()

    df["tipo"] = "imagem"

    if "sensibilidade_contaminada" not in df.columns and "recall_contaminada" in df.columns:
        df["sensibilidade_contaminada"] = df["recall_contaminada"]

    if "especificidade_nao_contaminada" not in df.columns:
        df["especificidade_nao_contaminada"] = df.apply(
            lambda linha: linha["tn"] / (linha["tn"] + linha["fp"]) if (linha["tn"] + linha["fp"]) else 0.0,
            axis=1,
        )

    return df


def gerar_conclusao(
    comparacao: pd.DataFrame,
    associacao: pd.DataFrame,
    taxas: pd.DataFrame,
) -> str:
    """Gera texto curto de conclusao do experimento."""
    metadata = comparacao[comparacao["tipo"] == "metadados"].copy()
    imagem = comparacao[comparacao["tipo"] == "imagem"].copy()

    melhor_metadata = metadata.sort_values("f1_contaminada", ascending=False).iloc[0]
    melhor_imagem = imagem.sort_values("f1_contaminada", ascending=False).iloc[0] if not imagem.empty else None

    maior_v = associacao["cramers_v"].max() if not associacao.empty else np.nan
    campo_maior_v = (
        associacao.sort_values("cramers_v", ascending=False).iloc[0]["campo"] if not associacao.empty else "indefinido"
    )

    indicio_forte = False
    motivos = []

    if not pd.isna(maior_v) and maior_v >= 0.30:
        indicio_forte = True
        motivos.append(f"ha associacao relevante entre metadados e contaminacao ({campo_maior_v}, Cramer's V={maior_v:.3f})")

    if melhor_imagem is not None:
        diff_f1 = float(melhor_metadata["f1_contaminada"] - melhor_imagem["f1_contaminada"])
        if diff_f1 >= -0.03:
            indicio_forte = True
            motivos.append(
                "o melhor modelo so com metadados ficou proximo ou acima do melhor modelo de imagem "
                f"(F1 metadados={melhor_metadata['f1_contaminada']:.3f}; "
                f"F1 imagem={melhor_imagem['f1_contaminada']:.3f})"
            )

    status = "INDICIO FORTE DE VIES DE LOTE/TRATAMENTO" if indicio_forte else "INDICIO FRACO OU INCONCLUSIVO"

    linhas = [
        status,
        "",
        "Resumo:",
        f"- Melhor modelo de metadados: {melhor_metadata['modelo']} / {melhor_metadata['cenario']}",
        f"  recall={melhor_metadata['recall_contaminada']:.3f}; "
        f"especificidade={melhor_metadata['especificidade_nao_contaminada']:.3f}; "
        f"F1={melhor_metadata['f1_contaminada']:.3f}.",
    ]

    if melhor_imagem is not None:
        linhas.extend(
            [
                f"- Melhor modelo de imagem: {melhor_imagem['modelo']} / {melhor_imagem['cenario']}",
                f"  recall={melhor_imagem['recall_contaminada']:.3f}; "
                f"especificidade={melhor_imagem['especificidade_nao_contaminada']:.3f}; "
                f"F1={melhor_imagem['f1_contaminada']:.3f}.",
            ]
        )

    if motivos:
        linhas.append("")
        linhas.append("Motivos:")
        for motivo in motivos:
            linhas.append(f"- {motivo}.")

    linhas.extend(
        [
            "",
            "Leitura correta:",
            "- Este resultado nao prova que o modelo de imagem esta errado.",
            "- Ele mostra que origem/tratamento/pasta carregam sinal suficiente para predizer contaminacao.",
            "- Portanto, parte do desempenho dos modelos de imagem pode estar vindo de vies de lote, tratamento, fundo, iluminacao ou organizacao das pastas.",
            "- A proxima avaliacao recomendada e separar treino/teste por experimento ou tratamento, para medir generalizacao fora do lote.",
        ]
    )

    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser()
    raiz_padrao = Path(__file__).resolve().parents[2]
    tabelas_padrao = raiz_padrao / "saidas" / "tabelas"
    saida_padrao = tabelas_padrao / "08_baseline_metadados"

    parser.add_argument("--tabelas-dir", type=Path, default=tabelas_padrao)
    parser.add_argument("--saida-dir", type=Path, default=saida_padrao)
    args = parser.parse_args()

    tabelas_dir = args.tabelas_dir
    saida_dir = args.saida_dir
    saida_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo tabelas de: {tabelas_dir}")
    df = preparar_tabela_com_split(tabelas_dir)

    X, colunas_categoricas, colunas_numericas = criar_features_metadados(df)
    y = df["contaminou"].astype(int)
    split = df["split"].astype(str)

    mascara_treino = split == "treino"
    mascara_validacao = split == "validacao"
    mascara_teste = split == "teste"

    preprocessador = criar_preprocessador(colunas_categoricas, colunas_numericas)

    modelos = {
        "metadata_logreg_balanced": Pipeline(
            steps=[
                ("prep", preprocessador),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "metadata_random_forest": Pipeline(
            steps=[
                ("prep", preprocessador),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=8,
                        random_state=42,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "metadata_gradient_boosting": Pipeline(
            steps=[
                ("prep", preprocessador),
                ("clf", GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.05, max_depth=2)),
            ]
        ),
    }

    resultados = []
    predicoes_teste = df.loc[mascara_teste, ["caminho_relativo", "nome_copiado", "classe", "contaminou", "split"]].copy()

    for nome, modelo in modelos.items():
        print(f"Treinando {nome}...")
        modelo.fit(X.loc[mascara_treino], y.loc[mascara_treino])

        prob_validacao = modelo.predict_proba(X.loc[mascara_validacao])[:, 1]
        prob_teste = modelo.predict_proba(X.loc[mascara_teste])[:, 1]

        resultados.extend(
            avaliar_modelo(
                nome_modelo=nome,
                tipo="metadados",
                y_validacao=y.loc[mascara_validacao],
                prob_validacao=prob_validacao,
                y_teste=y.loc[mascara_teste],
                prob_teste=prob_teste,
            )
        )

        predicoes_teste[f"prob_{nome}"] = prob_teste

    # Modelo de taxa historica por grupo.
    colunas_grupo = ["experimento_rotulo", "tratamento_planilha", "pasta_pai"]
    mapa_taxa, taxa_global = fit_prior_grupo(
        df.loc[mascara_treino],
        y.loc[mascara_treino],
        colunas_grupo=colunas_grupo,
        suavizacao=5.0,
    )
    prob_validacao = predict_prior_grupo(df.loc[mascara_validacao], mapa_taxa, taxa_global, colunas_grupo)
    prob_teste = predict_prior_grupo(df.loc[mascara_teste], mapa_taxa, taxa_global, colunas_grupo)

    resultados.extend(
        avaliar_modelo(
            nome_modelo="metadata_prior_grupo_experimento_tratamento_pasta",
            tipo="metadados",
            y_validacao=y.loc[mascara_validacao],
            prob_validacao=prob_validacao,
            y_teste=y.loc[mascara_teste],
            prob_teste=prob_teste,
        )
    )
    predicoes_teste["prob_metadata_prior_grupo_experimento_tratamento_pasta"] = prob_teste

    # Baseline de controle: classificar tudo como contaminada.
    linha_constante = metricas_binarias(y.loc[mascara_teste], np.ones(mascara_teste.sum()), 0.5)
    linha_constante["modelo"] = "baseline_constante_sempre_contaminada"
    linha_constante["tipo"] = "controle"
    linha_constante["cenario"] = "teste_regra_sempre_contaminada"
    resultados.append(linha_constante)

    resultados_df = pd.DataFrame(resultados)

    comparacao_imagem = carregar_comparacao_imagem(tabelas_dir)
    comparacao = pd.concat([comparacao_imagem, resultados_df], ignore_index=True, sort=False)

    colunas_comparacao = [
        "modelo",
        "tipo",
        "cenario",
        "threshold",
        "acuracia",
        "precisao_contaminada",
        "recall_contaminada",
        "sensibilidade_contaminada",
        "especificidade_nao_contaminada",
        "f1_contaminada",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    colunas_presentes = [c for c in colunas_comparacao if c in comparacao.columns]
    comparacao = comparacao[colunas_presentes].sort_values(
        ["tipo", "modelo", "cenario"],
        ascending=[True, True, True],
    )

    campos_associacao = [
        "experimento_rotulo",
        "tratamento_planilha",
        "pasta_esperada",
        "experimento_img",
        "pasta_pai",
        "origem_planilha",
        "id_letra",
        "qtd_observacoes",
        "largura",
        "altura",
    ]
    associacao = associacao_categorica(df.assign(id_letra=X["id_letra"]), campos_associacao)

    campos_taxas = ["experimento_rotulo", "tratamento_planilha", "pasta_pai"]
    taxas = taxas_por_grupo(df, campos_taxas)

    salvar_csv(comparacao, saida_dir / "comparacao_baseline_metadados_modelos.csv")
    salvar_csv(associacao, saida_dir / "associacao_metadados_contaminacao.csv")
    salvar_csv(taxas, saida_dir / "taxas_contaminacao_por_grupo.csv")
    salvar_csv(predicoes_teste, saida_dir / "predicoes_baseline_metadados_teste.csv")

    conclusao = gerar_conclusao(comparacao, associacao, taxas)
    (saida_dir / "conclusao_baseline_metadados.txt").write_text(conclusao, encoding="utf-8")

    print("\nArquivos gerados:")
    print(f"- {saida_dir / 'comparacao_baseline_metadados_modelos.csv'}")
    print(f"- {saida_dir / 'associacao_metadados_contaminacao.csv'}")
    print(f"- {saida_dir / 'taxas_contaminacao_por_grupo.csv'}")
    print(f"- {saida_dir / 'predicoes_baseline_metadados_teste.csv'}")
    print(f"- {saida_dir / 'conclusao_baseline_metadados.txt'}")
    print("\nConclusao:")
    print(conclusao)


if __name__ == "__main__":
    main()
