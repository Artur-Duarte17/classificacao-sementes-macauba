from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 34 - COMPARAR SCORES DA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Avaliar consenso oficial e estrategias individuais
# - Gerar micro/macro por estrategia
# - Marcar rankings externos como exploratorios, sem selecionar regra oficial
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TRIAGEM = PASTA_PROJETO / "saidas" / "tabelas" / "08_triagem"

CAMINHO_PREDICOES = PASTA_TRIAGEM / "predicoes_triagem_crossfit.csv"
CAMINHO_METRICAS_GRUPO = PASTA_TRIAGEM / "metricas_triagem_por_grupo.csv"
CAMINHO_RESUMO = PASTA_TRIAGEM / "resumo_triagem_micro_macro.csv"
CAMINHO_COMPARACAO = PASTA_TRIAGEM / "comparacao_scores_triagem.csv"
CAMINHO_RECOMENDADO = PASTA_TRIAGEM / "score_triagem_recomendado.csv"
CAMINHO_MANIFESTO = PASTA_TRIAGEM / "manifesto_comparacao_triagem.json"


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


def dividir(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def calcular_metricas(df: pd.DataFrame) -> dict:
    alvo = pd.to_numeric(df["alvo"], errors="coerce").astype(int)
    decisao = df["decisao_triagem"].astype(str)
    total = int(len(df))
    contaminadas = int(alvo.eq(1).sum())
    nao_contaminadas = int(alvo.eq(0).sum())

    baixo = decisao.eq("baixo_risco")
    alto = decisao.eq("alto_risco")
    incerto = decisao.eq("incerto")

    contaminadas_baixo = int((baixo & alvo.eq(1)).sum())
    nao_contaminadas_baixo = int((baixo & alvo.eq(0)).sum())
    contaminadas_alto = int((alto & alvo.eq(1)).sum())
    nao_contaminadas_alto = int((alto & alvo.eq(0)).sum())
    contaminadas_incerto = int((incerto & alvo.eq(1)).sum())
    nao_contaminadas_incerto = int((incerto & alvo.eq(0)).sum())

    n_baixo = int(baixo.sum())
    n_alto = int(alto.sum())
    n_incerto = int(incerto.sum())
    taxa_contaminada_baixo = dividir(contaminadas_baixo, n_baixo)
    seguranca_baixo = 1.0 - taxa_contaminada_baixo if n_baixo else np.nan
    recall_alto_contaminada = dividir(contaminadas_alto, contaminadas)
    utilidade_baixo_nao_contaminada = dividir(nao_contaminadas_baixo, nao_contaminadas)
    precisao_alto_contaminada = dividir(contaminadas_alto, n_alto)

    return {
        "total": total,
        "contaminadas": contaminadas,
        "nao_contaminadas": nao_contaminadas,
        "baixo_risco": n_baixo,
        "alto_risco": n_alto,
        "incerto": n_incerto,
        "contaminadas_baixo_risco": contaminadas_baixo,
        "nao_contaminadas_baixo_risco": nao_contaminadas_baixo,
        "contaminadas_alto_risco": contaminadas_alto,
        "nao_contaminadas_alto_risco": nao_contaminadas_alto,
        "contaminadas_incerto": contaminadas_incerto,
        "nao_contaminadas_incerto": nao_contaminadas_incerto,
        "taxa_baixo_risco": dividir(n_baixo, total),
        "taxa_alto_risco": dividir(n_alto, total),
        "taxa_incerto": dividir(n_incerto, total),
        "taxa_contaminada_baixo_risco": taxa_contaminada_baixo,
        "seguranca_baixo_risco": seguranca_baixo,
        "recall_alto_risco_contaminada": recall_alto_contaminada,
        "precisao_alto_risco_contaminada": precisao_alto_contaminada,
        "utilidade_baixo_risco_nao_contaminada": utilidade_baixo_nao_contaminada,
        "cobertura_decisao": dividir(n_baixo + n_alto, total),
    }


def metricas_por_grupo(predicoes: pd.DataFrame) -> pd.DataFrame:
    registros = []
    for (estrategia, grupo), dados in predicoes.groupby(["estrategia", "grupo_externo"]):
        primeira = dados.iloc[0]
        registros.append({
            "estrategia": estrategia,
            "grupo_externo": grupo,
            "tipo_estrategia": primeira.get("tipo_estrategia"),
            "estrategia_oficial": bool(primeira.get("estrategia_oficial", False)),
            "criterio_definido_antes_avaliacao": bool(
                primeira.get("criterio_definido_antes_avaliacao", True)
            ),
            "usa_resultado_externo_para_selecao": bool(
                primeira.get("usa_resultado_externo_para_selecao", False)
            ),
            **calcular_metricas(dados),
        })
    return pd.DataFrame(registros)


def resumo_micro_macro(predicoes: pd.DataFrame, metricas_grupo: pd.DataFrame) -> pd.DataFrame:
    registros = []
    metricas_media = [
        "taxa_baixo_risco",
        "taxa_alto_risco",
        "taxa_incerto",
        "taxa_contaminada_baixo_risco",
        "seguranca_baixo_risco",
        "recall_alto_risco_contaminada",
        "precisao_alto_risco_contaminada",
        "utilidade_baixo_risco_nao_contaminada",
        "cobertura_decisao",
    ]
    for estrategia, dados in predicoes.groupby("estrategia"):
        primeira = dados.iloc[0]
        registros.append({
            "agregacao": "micro",
            "estrategia": estrategia,
            "tipo_estrategia": primeira.get("tipo_estrategia"),
            "estrategia_oficial": bool(primeira.get("estrategia_oficial", False)),
            "comparacao_exploratoria": False,
            "usa_resultado_externo_para_selecao": False,
            **calcular_metricas(dados),
        })

    for estrategia, dados in metricas_grupo.groupby("estrategia"):
        primeira = dados.iloc[0]
        registro = {
            "agregacao": "macro",
            "estrategia": estrategia,
            "tipo_estrategia": primeira.get("tipo_estrategia"),
            "estrategia_oficial": bool(primeira.get("estrategia_oficial", False)),
            "comparacao_exploratoria": False,
            "usa_resultado_externo_para_selecao": False,
            "grupos": int(dados["grupo_externo"].nunique()),
        }
        for metrica in metricas_media:
            valores = pd.to_numeric(dados[metrica], errors="coerce")
            registro[f"{metrica}_media"] = float(valores.mean(skipna=True))
            registro[f"{metrica}_dp"] = float(valores.std(skipna=True, ddof=0))
        registros.append(registro)
    return pd.DataFrame(registros)


def criar_comparacao_exploratoria(resumo: pd.DataFrame) -> pd.DataFrame:
    micro = resumo[resumo["agregacao"].astype(str).eq("micro")].copy()
    micro["comparacao_exploratoria"] = True
    micro["nao_utilizada_para_ajuste_ou_selecao"] = True
    micro["usa_resultado_externo_para_selecao"] = False
    micro = micro.sort_values(
        [
            "contaminadas_baixo_risco",
            "recall_alto_risco_contaminada",
            "utilidade_baixo_risco_nao_contaminada",
            "cobertura_decisao",
        ],
        ascending=[True, False, False, False],
    )
    micro["ranking_exploratorio"] = range(1, len(micro) + 1)
    return micro


def criar_recomendado(resumo: pd.DataFrame) -> pd.DataFrame:
    oficial = resumo[
        (resumo["agregacao"].astype(str) == "micro")
        & (resumo["estrategia"].astype(str) == "consenso_pre_especificado")
    ].copy()
    if len(oficial) != 1:
        raise ValueError(
            "score_triagem_recomendado exige exatamente uma linha micro do consenso."
        )
    oficial["estrategia_oficial"] = "consenso_pre_especificado"
    oficial["criterio_definido_antes_avaliacao"] = True
    oficial["usa_resultado_externo_para_selecao"] = False
    oficial["comparacao_exploratoria"] = False
    oficial["nao_substituir_por_ranking_externo"] = True
    oficial["baixo_risco_nao_e_liberacao_automatica"] = True
    oficial["viabilidade_operacional"] = False
    oficial["motivo_inviabilidade"] = (
        "consenso_classificou_todas_amostras_como_alto_risco"
    )
    oficial["zona_baixo_risco_segura"] = False
    oficial["triagem_oficial_aprovada_para_uso"] = False
    oficial["resultado_cientifico"] = "triagem_nao_viavel_com_base_atual"
    return oficial


def main() -> None:
    print("=" * 70)
    print("COMPARANDO SCORES DA TRIAGEM")
    print("=" * 70)

    predicoes = ler_csv_obrigatorio(CAMINHO_PREDICOES)
    metricas_grupo = metricas_por_grupo(predicoes)
    resumo = resumo_micro_macro(predicoes, metricas_grupo)
    comparacao = criar_comparacao_exploratoria(resumo)
    recomendado = criar_recomendado(resumo)

    gravar_csv_atomico(metricas_grupo, CAMINHO_METRICAS_GRUPO)
    gravar_csv_atomico(resumo, CAMINHO_RESUMO)
    gravar_csv_atomico(comparacao, CAMINHO_COMPARACAO)
    gravar_csv_atomico(recomendado, CAMINHO_RECOMENDADO)
    gravar_json_atomico(
        {
            "protocolo": "triagem_preventiva_crossfit",
            "origem_predicoes": caminho_relativo(CAMINHO_PREDICOES),
            "estrategia_oficial": "consenso_pre_especificado",
            "criterio_definido_antes_avaliacao": True,
            "usa_resultado_externo_para_selecao": False,
            "comparacao_exploratoria": caminho_relativo(CAMINHO_COMPARACAO),
            "ranking_externo_nao_utilizado_para_selecao": True,
            "arquivos_saida": {
                "metricas_grupo": caminho_relativo(CAMINHO_METRICAS_GRUPO),
                "resumo": caminho_relativo(CAMINHO_RESUMO),
                "comparacao": caminho_relativo(CAMINHO_COMPARACAO),
                "recomendado": caminho_relativo(CAMINHO_RECOMENDADO),
            },
        },
        CAMINHO_MANIFESTO,
    )

    print(f"Estrategias avaliadas: {predicoes['estrategia'].nunique()}")
    print(f"- {CAMINHO_METRICAS_GRUPO}")
    print(f"- {CAMINHO_RESUMO}")
    print(f"- {CAMINHO_COMPARACAO}")
    print(f"- {CAMINHO_RECOMENDADO}")


if __name__ == "__main__":
    main()
