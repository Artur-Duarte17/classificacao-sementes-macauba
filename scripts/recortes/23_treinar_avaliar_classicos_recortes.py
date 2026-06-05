from pathlib import Path
import json
import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# SCRIPT 23 - TREINAR E AVALIAR CLASSICOS NOS RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Usar apenas atributos visuais extraidos dos recortes
# - Escolher hiperparametros somente por CV estratificada no treino
# - Usar validacao apenas para escolher thresholds
# - Avaliar o teste final uma unica vez
#
# Este script treina Random Forest e SVM RBF.
# Execute manualmente no ambiente conda.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_CLASSICOS = PASTA_TABELAS / "06_modelos" / "classicos"

CAMINHO_ATRIBUTOS = PASTA_CLASSICOS / "atributos_visuais_recortes.csv"
CAMINHO_METRICAS = PASTA_CLASSICOS / "metricas_classicos_teste.csv"
CAMINHO_PREDICOES = PASTA_CLASSICOS / "predicoes_classicos_teste.csv"
CAMINHO_THRESHOLDS = PASTA_CLASSICOS / "curva_threshold_classicos_validacao.csv"
CAMINHO_IMPORTANCIA_RF = PASTA_CLASSICOS / "importancia_random_forest.csv"
CAMINHO_PARAMETROS = PASTA_CLASSICOS / "melhores_parametros_classicos.csv"
CAMINHO_CV_RESULTADOS = PASTA_CLASSICOS / "cv_resultados_classicos.csv"
CAMINHO_FEATURES = PASTA_CLASSICOS / "features_classicos.csv"

CLASSE_POSITIVA = "contaminada"
INDICE_POSITIVO = 1
RECALL_MINIMO_PRIORITARIO = 0.95
SEMENTE_ALEATORIA = 42
CV_FOLDS = 5
N_JOBS_GRID = 6

CONJUNTO_PRINCIPAL = "principal_normalizado"
CONJUNTO_SENSIBILIDADE = "sensibilidade_todos_atributos"

COLUNAS_EXCLUIDAS_OBRIGATORIAS = {
    "nome_arquivo",
    "split",
    "classe",
    "classe_real",
    "alvo",
    "caminho_imagem",
    "caminho_recorte",
    "status_atributos",
    "erro_atributos",
}

COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL = {
    "largura_recorte",
    "altura_recorte",
    "pixels_recorte",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_area_px",
    "contour_area_px",
    "contour_perimeter_px",
}

PREFIXOS_TEXTURA_RESOLUCAO_ORIGINAL = (
    "texture_",
    "lbp_",
)

PREFIXOS_ATRIBUTOS_VISUAIS = (
    "rgb_",
    "hsv_",
    "brilho_",
    "contraste_",
    "mask_",
    "bbox_",
    "contour_",
    "circularity",
    "extent",
    "solidity",
    "hu_moment_",
    "texture_",
    "lbp_",
    "largura_recorte",
    "altura_recorte",
    "pixels_recorte",
)

warnings.filterwarnings("ignore", category=UserWarning)


def coluna_e_atributo_visual(coluna: str) -> bool:
    return coluna.startswith(PREFIXOS_ATRIBUTOS_VISUAIS)


def coluna_e_textura_original(coluna: str) -> bool:
    return coluna.startswith(PREFIXOS_TEXTURA_RESOLUCAO_ORIGINAL)


def ler_atributos() -> pd.DataFrame:
    if not CAMINHO_ATRIBUTOS.exists():
        raise FileNotFoundError(
            f"Atributos visuais nao encontrados: {CAMINHO_ATRIBUTOS}\n"
            "Execute antes: python scripts\\recortes\\22_extrair_atributos_visuais_recortes.py"
        )

    df = pd.read_csv(CAMINHO_ATRIBUTOS)
    colunas_obrigatorias = ["split", "classe", "alvo"]
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {faltantes}")

    if "status_atributos" in df.columns:
        erros = df[df["status_atributos"] != "ok"].copy()
        if not erros.empty:
            caminho_erros = PASTA_CLASSICOS / "registros_sem_atributos_classicos.csv"
            erros.to_csv(caminho_erros, index=False, encoding="utf-8-sig")
            raise ValueError(
                "Ha registros sem atributos visuais validos. "
                f"Conferir: {caminho_erros}"
            )

    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["classe"] = df["classe"].astype(str)
    df["alvo"] = pd.to_numeric(df["alvo"], errors="coerce").astype(int)

    splits_necessarios = {"treino", "validacao", "teste"}
    splits_encontrados = set(df["split"].unique())
    faltantes_split = sorted(splits_necessarios - splits_encontrados)
    if faltantes_split:
        raise ValueError(f"Splits obrigatorios ausentes: {faltantes_split}")

    return df


def obter_features_visuais(df: pd.DataFrame) -> list[str]:
    features = []
    for coluna in df.columns:
        if coluna in COLUNAS_EXCLUIDAS_OBRIGATORIAS:
            continue
        if not coluna_e_atributo_visual(coluna):
            continue

        valores = pd.to_numeric(df[coluna], errors="coerce")
        if valores.notna().sum() == 0:
            continue
        features.append(coluna)

    return sorted(features)


def obter_features_por_conjunto(df: pd.DataFrame) -> dict[str, list[str]]:
    todas = obter_features_visuais(df)
    principal = [
        coluna
        for coluna in todas
        if coluna not in COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL
        and not coluna_e_textura_original(coluna)
    ]

    if not principal:
        raise ValueError("Nenhuma feature disponivel no conjunto principal.")
    if not todas:
        raise ValueError("Nenhuma feature visual disponivel.")

    return {
        CONJUNTO_PRINCIPAL: principal,
        CONJUNTO_SENSIBILIDADE: todas,
    }


def preparar_matriz(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matriz = df[features].apply(pd.to_numeric, errors="coerce")
    matriz = matriz.replace([np.inf, -np.inf], np.nan)
    return matriz


def preparar_splits(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    treino = df[df["split"] == "treino"].copy()
    validacao = df[df["split"] == "validacao"].copy()
    teste = df[df["split"] == "teste"].copy()

    x_treino = preparar_matriz(treino, features)
    x_validacao = preparar_matriz(validacao, features)
    x_teste = preparar_matriz(teste, features)

    y_treino = treino["alvo"].to_numpy(dtype=int)
    y_validacao = validacao["alvo"].to_numpy(dtype=int)
    y_teste = teste["alvo"].to_numpy(dtype=int)

    return x_treino, y_treino, x_validacao, y_validacao, x_teste, y_teste


def validar_cv(y_treino: np.ndarray):
    contagens = np.bincount(y_treino, minlength=2)
    menor_classe = int(contagens.min())
    if menor_classe < CV_FOLDS:
        raise ValueError(
            f"CV estratificada de {CV_FOLDS} folds exige pelo menos "
            f"{CV_FOLDS} amostras por classe no treino. Contagens: {contagens.tolist()}"
        )


def criar_modelos() -> list[dict]:
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "modelo",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=SEMENTE_ALEATORIA,
                n_jobs=1,
            ),
        ),
    ])
    svm_rbf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "modelo",
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEMENTE_ALEATORIA,
            ),
        ),
    ])

    return [
        {
            "modelo": "random_forest",
            "estimador": random_forest,
            "grid": {
                "modelo__n_estimators": [500, 1000],
                "modelo__max_depth": [None, 8, 16],
                "modelo__min_samples_leaf": [1, 3, 5],
                "modelo__min_samples_split": [2, 5, 10],
                "modelo__max_features": ["sqrt", "log2"],
            },
        },
        {
            "modelo": "svm_rbf",
            "estimador": svm_rbf,
            "grid": {
                "modelo__C": [0.1, 1, 3, 10, 30, 100],
                "modelo__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
            },
        },
    ]


def criar_scoring() -> dict:
    return {
        "f1_contaminada": make_scorer(
            f1_score,
            pos_label=INDICE_POSITIVO,
            zero_division=0,
        ),
        "recall_contaminada": make_scorer(
            recall_score,
            pos_label=INDICE_POSITIVO,
            zero_division=0,
        ),
        "precisao_contaminada": make_scorer(
            precision_score,
            pos_label=INDICE_POSITIVO,
            zero_division=0,
        ),
        "especificidade_nao_contaminada": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        ),
        "acuracia": make_scorer(accuracy_score),
    }


def selecionar_melhor_cv(cv_resultados: pd.DataFrame) -> pd.Series:
    ordenado = cv_resultados.sort_values(
        [
            "mean_test_f1_contaminada",
            "mean_test_recall_contaminada",
            "mean_test_especificidade_nao_contaminada",
            "mean_test_acuracia",
            "std_test_f1_contaminada",
        ],
        ascending=[False, False, False, False, True],
    )
    return ordenado.iloc[0]


def parametros_sem_prefixo(parametros: dict) -> dict:
    saida = {}
    for chave, valor in parametros.items():
        saida[chave.replace("modelo__", "")] = valor
    return saida


def buscar_hiperparametros(
    nome_modelo: str,
    estimador,
    grid: dict,
    x_treino: pd.DataFrame,
    y_treino: np.ndarray,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )
    busca = GridSearchCV(
        estimator=estimador,
        param_grid=grid,
        scoring=criar_scoring(),
        refit=False,
        cv=cv,
        n_jobs=N_JOBS_GRID,
        pre_dispatch=N_JOBS_GRID,
        verbose=2,
        return_train_score=False,
    )
    busca.fit(x_treino, y_treino)

    resultados = pd.DataFrame(busca.cv_results_)
    melhor = selecionar_melhor_cv(resultados)
    melhor_parametros = dict(melhor["params"])

    colunas_resultados = [
        "params",
        "mean_test_f1_contaminada",
        "std_test_f1_contaminada",
        "mean_test_recall_contaminada",
        "std_test_recall_contaminada",
        "mean_test_especificidade_nao_contaminada",
        "std_test_especificidade_nao_contaminada",
        "mean_test_precisao_contaminada",
        "mean_test_acuracia",
    ]
    colunas_resultados = [coluna for coluna in colunas_resultados if coluna in resultados.columns]
    resultados = resultados[colunas_resultados].copy()
    resultados["params_json"] = resultados["params"].map(
        lambda item: json.dumps(parametros_sem_prefixo(dict(item)), sort_keys=True)
    )
    resultados = resultados.drop(columns=["params"])
    resultados.insert(0, "modelo", nome_modelo)

    return melhor_parametros, resultados, melhor


def calcular_metricas(y_real, prob_contaminada, threshold: float) -> dict:
    y_real = np.asarray(y_real).astype(int)
    prob_contaminada = np.asarray(prob_contaminada).astype(float)
    pred = (prob_contaminada >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    precisao = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    especificidade = tn / max(tn + fp, 1)
    f1 = 2 * precisao * recall / max(precisao + recall, 1e-12)

    return {
        "threshold": float(threshold),
        "acuracia": float(accuracy_score(y_real, pred)),
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


def gerar_curva_threshold(y_validacao, prob_validacao) -> pd.DataFrame:
    registros = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        registros.append(
            calcular_metricas(
                y_validacao,
                prob_validacao,
                threshold=round(float(threshold), 2),
            )
        )
    return pd.DataFrame(registros)


def escolher_threshold_por_f1(df_thresholds: pd.DataFrame) -> float:
    melhor = df_thresholds.sort_values(
        ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(melhor["threshold"])


def escolher_threshold_por_recall(df_thresholds: pd.DataFrame) -> float:
    candidatos = df_thresholds[
        df_thresholds["recall_contaminada"] >= RECALL_MINIMO_PRIORITARIO
    ].copy()

    if candidatos.empty:
        candidatos = df_thresholds.copy()
        ordenacao = [
            "recall_contaminada",
            "f1_contaminada",
            "especificidade_nao_contaminada",
        ]
    else:
        ordenacao = [
            "f1_contaminada",
            "recall_contaminada",
            "especificidade_nao_contaminada",
        ]

    melhor = candidatos.sort_values(ordenacao, ascending=[False, False, False]).iloc[0]
    return float(melhor["threshold"])


def obter_probabilidade_contaminada(modelo, x: pd.DataFrame) -> np.ndarray:
    probabilidades = modelo.predict_proba(x)
    return probabilidades[:, INDICE_POSITIVO]


def adicionar_predicoes_threshold(
    df: pd.DataFrame,
    prob_coluna: str,
    sufixo: str,
    threshold: float,
) -> pd.DataFrame:
    coluna = f"predito_threshold_{sufixo}"
    df[coluna] = np.where(
        df[prob_coluna] >= threshold,
        "contaminada",
        "nao_contaminada",
    )
    return df


def gerar_predicoes_teste(
    df_base_teste: pd.DataFrame,
    conjunto_features: str,
    nome_modelo: str,
    prob_teste: np.ndarray,
    threshold_f1: float,
    threshold_recall: float,
) -> pd.DataFrame:
    colunas_auditoria = [
        "nome_arquivo",
        "split",
        "classe",
        "classe_real",
        "alvo",
        "caminho_recorte",
    ]
    colunas_auditoria = [coluna for coluna in colunas_auditoria if coluna in df_base_teste.columns]

    predicoes = df_base_teste[colunas_auditoria].copy()
    predicoes.insert(0, "modelo", nome_modelo)
    predicoes.insert(0, "conjunto_features", conjunto_features)
    predicoes["prob_contaminada"] = prob_teste
    predicoes = adicionar_predicoes_threshold(predicoes, "prob_contaminada", "0_50", 0.50)
    predicoes = adicionar_predicoes_threshold(
        predicoes,
        "prob_contaminada",
        "melhor_f1_validacao",
        threshold_f1,
    )
    predicoes = adicionar_predicoes_threshold(
        predicoes,
        "prob_contaminada",
        "prioridade_recall_validacao",
        threshold_recall,
    )
    return predicoes


def extrair_importancias_rf(
    modelo_treinado,
    features: list[str],
    conjunto_features: str,
    nome_modelo: str,
) -> pd.DataFrame:
    if nome_modelo != "random_forest":
        return pd.DataFrame()

    rf = modelo_treinado.named_steps["modelo"]
    importancias = pd.DataFrame({
        "conjunto_features": conjunto_features,
        "modelo": nome_modelo,
        "feature": features,
        "importancia": rf.feature_importances_,
    })
    return importancias.sort_values(
        ["conjunto_features", "importancia"],
        ascending=[True, False],
    )


def registrar_features(conjuntos: dict[str, list[str]]) -> pd.DataFrame:
    registros = []
    todas = set(conjuntos[CONJUNTO_SENSIBILIDADE])
    principal = set(conjuntos[CONJUNTO_PRINCIPAL])

    for conjunto, features in conjuntos.items():
        for feature in features:
            registros.append({
                "conjunto_features": conjunto,
                "feature": feature,
                "incluida_no_principal": feature in principal,
                "incluida_na_sensibilidade": feature in todas,
                "excluida_do_principal_por_absoluta": (
                    feature in COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL
                ),
                "excluida_do_principal_por_textura_original": (
                    coluna_e_textura_original(feature)
                ),
            })

    return pd.DataFrame(registros)


def main():
    print("=" * 60)
    print("TREINANDO CLASSICOS COM ATRIBUTOS VISUAIS DOS RECORTES")
    print("=" * 60)

    PASTA_CLASSICOS.mkdir(parents=True, exist_ok=True)

    df = ler_atributos()
    conjuntos_features = obter_features_por_conjunto(df)
    registrar_features(conjuntos_features).to_csv(
        CAMINHO_FEATURES,
        index=False,
        encoding="utf-8-sig",
    )

    print("Registros por split e classe:")
    print(pd.crosstab(df["split"], df["classe"]).to_string())
    print()
    print("Features por conjunto:")
    for conjunto, features in conjuntos_features.items():
        print(f"- {conjunto}: {len(features)}")

    metricas_todas = []
    predicoes_todas = []
    thresholds_todos = []
    parametros_todos = []
    cv_resultados_todos = []
    importancias_todas = []

    modelos = criar_modelos()

    for conjunto_features, features in conjuntos_features.items():
        print()
        print("=" * 60)
        print(f"Conjunto de features: {conjunto_features} ({len(features)} features)")
        print("=" * 60)

        x_treino, y_treino, x_validacao, y_validacao, x_teste, y_teste = preparar_splits(
            df,
            features,
        )
        validar_cv(y_treino)
        df_teste = df[df["split"] == "teste"].copy()

        for item_modelo in modelos:
            nome_modelo = item_modelo["modelo"]
            print()
            print(f"Buscando hiperparametros por CV: {nome_modelo}")

            melhor_parametros, cv_resultados, melhor_cv = buscar_hiperparametros(
                nome_modelo,
                item_modelo["estimador"],
                item_modelo["grid"],
                x_treino,
                y_treino,
            )
            cv_resultados.insert(0, "conjunto_features", conjunto_features)
            cv_resultados_todos.append(cv_resultados)

            modelo_treinado = clone(item_modelo["estimador"])
            modelo_treinado.set_params(**melhor_parametros)
            modelo_treinado.fit(x_treino, y_treino)

            prob_validacao = obter_probabilidade_contaminada(modelo_treinado, x_validacao)
            prob_teste = obter_probabilidade_contaminada(modelo_treinado, x_teste)

            thresholds = gerar_curva_threshold(y_validacao, prob_validacao)
            thresholds.insert(0, "modelo", nome_modelo)
            thresholds.insert(0, "conjunto_features", conjunto_features)
            thresholds_todos.append(thresholds)

            threshold_f1 = escolher_threshold_por_f1(thresholds)
            threshold_recall = escolher_threshold_por_recall(thresholds)

            cenarios = [
                ("teste_threshold_0_50", 0.50),
                ("teste_threshold_melhor_f1_validacao", threshold_f1),
                ("teste_threshold_prioridade_recall_validacao", threshold_recall),
            ]

            for cenario, threshold in cenarios:
                metricas = calcular_metricas(y_teste, prob_teste, threshold)
                metricas_todas.append({
                    "conjunto_features": conjunto_features,
                    "modelo": nome_modelo,
                    "cenario": cenario,
                    "n_features": len(features),
                    "inclui_features_absolutas": (
                        conjunto_features == CONJUNTO_SENSIBILIDADE
                    ),
                    "inclui_textura_resolucao_original": (
                        conjunto_features == CONJUNTO_SENSIBILIDADE
                    ),
                    **metricas,
                })

            predicoes_todas.append(
                gerar_predicoes_teste(
                    df_teste,
                    conjunto_features,
                    nome_modelo,
                    prob_teste,
                    threshold_f1,
                    threshold_recall,
                )
            )

            parametros_todos.append({
                "conjunto_features": conjunto_features,
                "modelo": nome_modelo,
                "n_features": len(features),
                "cv_folds": CV_FOLDS,
                "criterio_hiperparametros": (
                    "cv_treino_f1_contaminada_desempate_recall_especificidade"
                ),
                "mean_cv_f1_contaminada": float(melhor_cv["mean_test_f1_contaminada"]),
                "mean_cv_recall_contaminada": float(
                    melhor_cv["mean_test_recall_contaminada"]
                ),
                "mean_cv_especificidade_nao_contaminada": float(
                    melhor_cv["mean_test_especificidade_nao_contaminada"]
                ),
                "mean_cv_acuracia": float(melhor_cv["mean_test_acuracia"]),
                "threshold_melhor_f1_validacao": threshold_f1,
                "threshold_prioridade_recall_validacao": threshold_recall,
                "parametros_json": json.dumps(
                    parametros_sem_prefixo(melhor_parametros),
                    sort_keys=True,
                ),
            })

            importancias = extrair_importancias_rf(
                modelo_treinado,
                features,
                conjunto_features,
                nome_modelo,
            )
            if not importancias.empty:
                importancias_todas.append(importancias)

    metricas = pd.DataFrame(metricas_todas)
    predicoes = pd.concat(predicoes_todas, ignore_index=True)
    thresholds = pd.concat(thresholds_todos, ignore_index=True)
    parametros = pd.DataFrame(parametros_todos)
    cv_resultados = pd.concat(cv_resultados_todos, ignore_index=True)
    importancias_rf = (
        pd.concat(importancias_todas, ignore_index=True)
        if importancias_todas
        else pd.DataFrame()
    )

    metricas.to_csv(CAMINHO_METRICAS, index=False, encoding="utf-8-sig")
    predicoes.to_csv(CAMINHO_PREDICOES, index=False, encoding="utf-8-sig")
    thresholds.to_csv(CAMINHO_THRESHOLDS, index=False, encoding="utf-8-sig")
    parametros.to_csv(CAMINHO_PARAMETROS, index=False, encoding="utf-8-sig")
    cv_resultados.to_csv(CAMINHO_CV_RESULTADOS, index=False, encoding="utf-8-sig")
    importancias_rf.to_csv(CAMINHO_IMPORTANCIA_RF, index=False, encoding="utf-8-sig")

    print()
    print("Metricas no teste:")
    print(metricas.to_string(index=False))
    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_METRICAS,
        CAMINHO_PREDICOES,
        CAMINHO_THRESHOLDS,
        CAMINHO_PARAMETROS,
        CAMINHO_CV_RESULTADOS,
        CAMINHO_IMPORTANCIA_RF,
        CAMINHO_FEATURES,
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
