from pathlib import Path
from math import ceil
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 33 - CALIBRAR THRESHOLDS CROSSFIT DA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Derivar thresholds baixo/alto por modelo e grupo externo usando
#   somente a validacao interna do proprio fold
# - Aplicar a regra congelada no grupo externo
# - Gerar consenso oficial pre-especificado
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TRIAGEM = PASTA_PROJETO / "saidas" / "tabelas" / "08_triagem"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM / "tabela_integrada_triagem.csv"
CAMINHO_THRESHOLDS_INTERNOS = PASTA_TRIAGEM / "thresholds_internos_modelos_triagem.csv"

CAMINHO_THRESHOLDS_CROSSFIT = PASTA_TRIAGEM / "thresholds_crossfit_por_grupo.csv"
CAMINHO_PREDICOES_CROSSFIT = PASTA_TRIAGEM / "predicoes_triagem_crossfit.csv"
CAMINHO_CASOS_CRITICOS = PASTA_TRIAGEM / "casos_criticos_triagem.csv"
CAMINHO_MANIFESTO = PASTA_TRIAGEM / "manifesto_thresholds_triagem.json"


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio ausente: {caminho}")
    df = pd.read_csv(caminho)
    if df.empty:
        raise ValueError(f"Arquivo obrigatorio vazio: {caminho}")
    return df


def gravar_csv_atomico(df: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f"{caminho.name}.tmp")
    df.to_csv(temporario, index=False, encoding="utf-8-sig")
    pd.read_csv(temporario)
    temporario.replace(caminho)


def gravar_json_atomico(objeto: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f"{caminho.name}.tmp")
    temporario.write_text(
        json.dumps(objeto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    json.loads(temporario.read_text(encoding="utf-8"))
    temporario.replace(caminho)


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def slug_modelo(modelo: str, conjunto_features: str) -> str:
    texto = f"{modelo}_{conjunto_features}"
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_").lower()
    return texto


def colunas_probabilidade(tabela: pd.DataFrame) -> list[str]:
    return sorted(coluna for coluna in tabela.columns if coluna.startswith("prob_"))


def escolher_threshold_alto(curva: pd.DataFrame) -> pd.Series:
    dados = curva.copy()
    for coluna in ["threshold", "f1_contaminada", "recall_contaminada", "precisao_contaminada", "fp"]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados = dados.dropna(subset=["threshold", "f1_contaminada"])
    if dados.empty:
        raise ValueError("Curva de thresholds sem candidatos para threshold alto.")
    return dados.sort_values(
        [
            "f1_contaminada",
            "recall_contaminada",
            "precisao_contaminada",
            "fp",
            "threshold",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def calcular_minimo_utilidade(curva: pd.DataFrame) -> tuple[int, int]:
    suporte = pd.to_numeric(
        curva["suporte_nao_contaminada"],
        errors="coerce",
    ).dropna()
    if suporte.empty:
        raise ValueError("Curva de thresholds sem suporte_nao_contaminada.")
    suporte_unico = sorted(suporte.astype(int).unique())
    if len(suporte_unico) != 1:
        raise ValueError(
            "Curva de thresholds deveria ter suporte_nao_contaminada unico; "
            f"encontrado: {suporte_unico}"
        )
    suporte_validacao = int(suporte_unico[0])
    return suporte_validacao, int(max(5, ceil(0.05 * suporte_validacao)))


def escolher_threshold_baixo(
    curva: pd.DataFrame,
    threshold_alto: float,
    minimo_utilidade: int,
) -> tuple[float | None, str]:
    dados = curva.copy()
    for coluna in ["threshold", "fn", "tn"]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    candidatos = dados[
        dados["fn"].eq(0)
        & (dados["tn"] >= int(minimo_utilidade))
        & (dados["threshold"] < float(threshold_alto))
    ].copy()
    if candidatos.empty:
        return None, "sem_threshold_baixo_seguro"
    melhor = candidatos.sort_values("threshold", ascending=False).iloc[0]
    return float(melhor["threshold"]), "ok"


def derivar_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    registros = []
    for (fold, grupo, modelo, conjunto), curva in thresholds.groupby(
        ["fold", "grupo_externo", "modelo", "conjunto_features"]
    ):
        alto = escolher_threshold_alto(curva)
        threshold_alto = float(alto["threshold"])
        suporte_nao_contaminada, minimo_utilidade = calcular_minimo_utilidade(curva)
        threshold_baixo, status_baixo = escolher_threshold_baixo(
            curva,
            threshold_alto,
            minimo_utilidade,
        )
        baixo_valido = threshold_baixo is not None and threshold_baixo < threshold_alto
        if not baixo_valido:
            threshold_baixo = np.nan

        registros.append({
            "fold": int(fold),
            "grupo_externo": grupo,
            "modelo": modelo,
            "conjunto_features": conjunto,
            "modelo_slug": slug_modelo(str(modelo), str(conjunto)),
            "threshold_baixo": threshold_baixo,
            "threshold_alto": threshold_alto,
            "baixo_risco_disponivel": bool(baixo_valido),
            "status_threshold_baixo": status_baixo if baixo_valido else "baixo_risco_suspenso",
            "criterio_threshold_baixo": (
                "maior_threshold_com_fn_0_tn_minimo_e_menor_que_threshold_alto"
            ),
            "criterio_threshold_alto": (
                "melhor_f1_desempate_recall_precisao_menor_fp"
            ),
            "validacao_tn_threshold_alto": int(alto["tn"]),
            "validacao_fp_threshold_alto": int(alto["fp"]),
            "validacao_fn_threshold_alto": int(alto["fn"]),
            "validacao_tp_threshold_alto": int(alto["tp"]),
            "validacao_f1_threshold_alto": float(alto["f1_contaminada"]),
            "validacao_recall_threshold_alto": float(alto["recall_contaminada"]),
            "validacao_especificidade_threshold_alto": float(
                alto["especificidade_nao_contaminada"]
            ),
            "suporte_nao_contaminada_validacao": suporte_nao_contaminada,
            "minimo_utilidade_baixo_risco": minimo_utilidade,
            "formula_minimo_utilidade": "max(5, ceil(0.05 * suporte_nao_contaminada_validacao))",
            "origem_thresholds": "validacao_interna_mesmo_fold",
            "usa_resultado_externo_para_selecao": False,
        })
    return pd.DataFrame(registros)


def classificar_individual(score: float, threshold_baixo, threshold_alto: float) -> str:
    if pd.notna(threshold_baixo) and float(score) < float(threshold_baixo):
        return "baixo_risco"
    if float(score) >= float(threshold_alto):
        return "alto_risco"
    return "incerto"


def aplicar_individuais(tabela: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    registros = []
    colunas_base = [
        "fold",
        "grupo_externo",
        "grupo_validacao",
        "nome_arquivo",
        "caminho_relativo",
        "classe_real",
        "alvo",
        "split_original",
        "experimento_tratamento",
    ]
    for _, regra in thresholds.iterrows():
        coluna_prob = f"prob_{regra['modelo_slug']}"
        if coluna_prob not in tabela.columns:
            continue
        dados_fold = tabela[tabela["fold"].astype(int).eq(int(regra["fold"]))].copy()
        for _, linha in dados_fold.iterrows():
            score = float(linha[coluna_prob])
            decisao = classificar_individual(
                score,
                regra["threshold_baixo"],
                float(regra["threshold_alto"]),
            )
            registro = {coluna: linha[coluna] for coluna in colunas_base}
            registro.update({
                "estrategia": f"individual_{regra['modelo_slug']}",
                "tipo_estrategia": "individual_descritiva",
                "modelo": regra["modelo"],
                "conjunto_features": regra["conjunto_features"],
                "score_contaminacao": score,
                "threshold_baixo": regra["threshold_baixo"],
                "threshold_alto": regra["threshold_alto"],
                "baixo_risco_disponivel": bool(regra["baixo_risco_disponivel"]),
                "decisao_triagem": decisao,
                "estrategia_oficial": False,
                "criterio_definido_antes_avaliacao": True,
                "usa_resultado_externo_para_selecao": False,
            })
            registros.append(registro)
    return pd.DataFrame(registros)


def aplicar_consenso(tabela: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    registros = []
    colunas_base = [
        "fold",
        "grupo_externo",
        "grupo_validacao",
        "nome_arquivo",
        "caminho_relativo",
        "classe_real",
        "alvo",
        "split_original",
        "experimento_tratamento",
    ]
    for fold, regras_fold in thresholds.groupby("fold"):
        dados_fold = tabela[tabela["fold"].astype(int).eq(int(fold))].copy()
        regras_fold = regras_fold.copy()
        baixo_disponivel_fold = bool(regras_fold["baixo_risco_disponivel"].all())
        for _, linha in dados_fold.iterrows():
            modelos_alto = []
            modelos_baixo = []
            for _, regra in regras_fold.iterrows():
                coluna_prob = f"prob_{regra['modelo_slug']}"
                score = float(linha[coluna_prob])
                if score >= float(regra["threshold_alto"]):
                    modelos_alto.append(regra["modelo_slug"])
                if (
                    pd.notna(regra["threshold_baixo"])
                    and score < float(regra["threshold_baixo"])
                ):
                    modelos_baixo.append(regra["modelo_slug"])

            if modelos_alto:
                decisao = "alto_risco"
            elif baixo_disponivel_fold and len(modelos_baixo) == len(regras_fold):
                decisao = "baixo_risco"
            else:
                decisao = "incerto"

            registro = {coluna: linha[coluna] for coluna in colunas_base}
            registro.update({
                "estrategia": "consenso_pre_especificado",
                "tipo_estrategia": "consenso_oficial",
                "modelo": "consenso_modelos_visuais",
                "conjunto_features": "modelos_visuais_completos",
                "score_contaminacao": np.nan,
                "threshold_baixo": np.nan,
                "threshold_alto": np.nan,
                "baixo_risco_disponivel": baixo_disponivel_fold,
                "decisao_triagem": decisao,
                "modelos_acima_threshold_alto": ",".join(modelos_alto),
                "modelos_abaixo_threshold_baixo": ",".join(modelos_baixo),
                "n_modelos_consenso": int(len(regras_fold)),
                "estrategia_oficial": True,
                "criterio_definido_antes_avaliacao": True,
                "usa_resultado_externo_para_selecao": False,
            })
            registros.append(registro)
    return pd.DataFrame(registros)


def identificar_casos_criticos(predicoes: pd.DataFrame) -> pd.DataFrame:
    criticos = predicoes[
        (
            predicoes["decisao_triagem"].eq("baixo_risco")
            & pd.to_numeric(predicoes["alvo"], errors="coerce").eq(1)
        )
        | (
            predicoes["decisao_triagem"].eq("alto_risco")
            & pd.to_numeric(predicoes["alvo"], errors="coerce").eq(0)
        )
    ].copy()
    if criticos.empty:
        return pd.DataFrame(columns=[*predicoes.columns, "tipo_caso_critico"])
    criticos["tipo_caso_critico"] = np.where(
        criticos["decisao_triagem"].eq("baixo_risco"),
        "contaminada_em_baixo_risco",
        "nao_contaminada_em_alto_risco",
    )
    return criticos


def main() -> None:
    print("=" * 70)
    print("CALIBRANDO THRESHOLDS CROSSFIT DA TRIAGEM")
    print("=" * 70)

    tabela = ler_csv_obrigatorio(CAMINHO_TABELA_INTEGRADA)
    thresholds_internos = ler_csv_obrigatorio(CAMINHO_THRESHOLDS_INTERNOS)

    thresholds = derivar_thresholds(thresholds_internos)
    pred_individuais = aplicar_individuais(tabela, thresholds)
    pred_consenso = aplicar_consenso(tabela, thresholds)
    predicoes = pd.concat([pred_individuais, pred_consenso], ignore_index=True)
    casos_criticos = identificar_casos_criticos(predicoes)

    gravar_csv_atomico(thresholds, CAMINHO_THRESHOLDS_CROSSFIT)
    gravar_csv_atomico(predicoes, CAMINHO_PREDICOES_CROSSFIT)
    gravar_csv_atomico(casos_criticos, CAMINHO_CASOS_CRITICOS)
    gravar_json_atomico(
        {
            "protocolo": "triagem_preventiva_crossfit",
            "origem_tabela_integrada": caminho_relativo(CAMINHO_TABELA_INTEGRADA),
            "origem_thresholds_internos": caminho_relativo(CAMINHO_THRESHOLDS_INTERNOS),
            "threshold_baixo": {
                "criterio": "maior_threshold_com_fn_0_tn_minimo_e_menor_que_threshold_alto",
                "formula_minimo_utilidade": "max(5, ceil(0.05 * suporte_nao_contaminada_validacao))",
                "coluna_minimo_utilidade": "minimo_utilidade_baixo_risco",
                "minimos_utilidade_por_modelo_fold": json.loads(
                    thresholds[
                        [
                            "fold",
                            "grupo_externo",
                            "modelo",
                            "conjunto_features",
                            "suporte_nao_contaminada_validacao",
                            "minimo_utilidade_baixo_risco",
                        ]
                    ].to_json(orient="records", force_ascii=False)
                ),
                "sem_candidato": "nao_existe_zona_de_baixo_risco_modelo_fold",
            },
            "threshold_alto": {
                "criterio": "melhor_f1_desempate_recall_precisao_menor_fp",
            },
            "estrategia_oficial": "consenso_pre_especificado",
            "criterio_definido_antes_avaliacao": True,
            "usa_resultado_externo_para_selecao": False,
            "arquivos_saida": {
                "thresholds": caminho_relativo(CAMINHO_THRESHOLDS_CROSSFIT),
                "predicoes": caminho_relativo(CAMINHO_PREDICOES_CROSSFIT),
                "casos_criticos": caminho_relativo(CAMINHO_CASOS_CRITICOS),
            },
        },
        CAMINHO_MANIFESTO,
    )

    print(f"Thresholds gerados: {len(thresholds)}")
    print(f"Predicoes de triagem: {len(predicoes)}")
    print(f"Casos criticos: {len(casos_criticos)}")
    print(f"- {CAMINHO_THRESHOLDS_CROSSFIT}")
    print(f"- {CAMINHO_PREDICOES_CROSSFIT}")
    print(f"- {CAMINHO_CASOS_CRITICOS}")


if __name__ == "__main__":
    main()
