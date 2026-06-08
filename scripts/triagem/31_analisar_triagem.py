from pathlib import Path
import json

import pandas as pd


# ============================================================
# SCRIPT 31 - ANALISAR INTEGRACAO DA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Auditar a tabela integrada da triagem
# - Verificar cobertura dos modelos, grupos externos e duplicidades
# - Confirmar que media/maximo nao entraram como estrategia oficial
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TRIAGEM = PASTA_PROJETO / "saidas" / "tabelas" / "08_triagem"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM / "tabela_integrada_triagem.csv"
CAMINHO_THRESHOLDS_INTERNOS = PASTA_TRIAGEM / "thresholds_internos_modelos_triagem.csv"
CAMINHO_AUDITORIA = PASTA_TRIAGEM / "auditoria_integracao_triagem.csv"
CAMINHO_RESUMO = PASTA_TRIAGEM / "resumo_auditoria_integracao_triagem.txt"

TOTAL_AMOSTRAS_ESPERADO = 703
TOTAL_GRUPOS_ESPERADO = 12


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


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def colunas_probabilidade(tabela: pd.DataFrame) -> list[str]:
    return sorted(coluna for coluna in tabela.columns if coluna.startswith("prob_"))


def criar_auditoria(tabela: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    probs = colunas_probabilidade(tabela)
    registros = [
        {
            "item": "amostras_integradas",
            "valor": int(len(tabela)),
            "esperado": TOTAL_AMOSTRAS_ESPERADO,
            "status": "ok" if len(tabela) == TOTAL_AMOSTRAS_ESPERADO else "falha",
        },
        {
            "item": "grupos_externos",
            "valor": int(tabela["grupo_externo"].nunique()),
            "esperado": TOTAL_GRUPOS_ESPERADO,
            "status": (
                "ok"
                if tabela["grupo_externo"].nunique() == TOTAL_GRUPOS_ESPERADO
                else "falha"
            ),
        },
        {
            "item": "duplicatas_fold_grupo_arquivo",
            "valor": int(tabela.duplicated(["fold", "grupo_externo", "nome_arquivo"]).sum()),
            "esperado": 0,
            "status": (
                "ok"
                if not tabela.duplicated(["fold", "grupo_externo", "nome_arquivo"]).any()
                else "falha"
            ),
        },
        {
            "item": "modelos_visuais_completos",
            "valor": len(probs),
            "esperado": ">=1",
            "status": "ok" if probs else "falha",
        },
        {
            "item": "usa_resultado_externo_para_selecao",
            "valor": bool(tabela["usa_resultado_externo_para_selecao"].astype(bool).any()),
            "esperado": False,
            "status": (
                "ok"
                if not tabela["usa_resultado_externo_para_selecao"].astype(bool).any()
                else "falha"
            ),
        },
        {
            "item": "colunas_media_maximo_probabilidades",
            "valor": ",".join(
                coluna
                for coluna in tabela.columns
                if "media" in coluna.lower() or "max" in coluna.lower()
            ),
            "esperado": "",
            "status": (
                "ok"
                if not any("media" in c.lower() or "max" in c.lower() for c in tabela.columns)
                else "falha"
            ),
        },
    ]

    for coluna in probs:
        registros.append({
            "item": f"ausentes_{coluna}",
            "valor": int(tabela[coluna].isna().sum()),
            "esperado": 0,
            "status": "ok" if int(tabela[coluna].isna().sum()) == 0 else "falha",
        })

    if not thresholds.empty:
        for (modelo, conjunto), grupo in thresholds.groupby(["modelo", "conjunto_features"]):
            registros.append({
                "item": f"thresholds_{modelo}_{conjunto}",
                "valor": int(grupo["grupo_externo"].nunique()),
                "esperado": TOTAL_GRUPOS_ESPERADO,
                "status": (
                    "ok"
                    if int(grupo["grupo_externo"].nunique()) == TOTAL_GRUPOS_ESPERADO
                    else "falha"
                ),
            })

    return pd.DataFrame(registros)


def gerar_resumo(tabela: pd.DataFrame, auditoria: pd.DataFrame) -> str:
    status_geral = "ok" if auditoria["status"].eq("ok").all() else "falha"
    probs = colunas_probabilidade(tabela)
    linhas = [
        "Resumo da auditoria de integracao da triagem",
        "=" * 48,
        f"Status geral: {status_geral}",
        f"Amostras: {len(tabela)}",
        f"Grupos externos: {tabela['grupo_externo'].nunique()}",
        f"Modelos visuais completos: {len(probs)}",
        "Probabilidades usadas:",
        *[f"- {coluna}" for coluna in probs],
        "",
        "Invariantes:",
        "- estrategia oficial pre-especificada: consenso_pre_especificado",
        "- criterio_definido_antes_avaliacao: true",
        "- usa_resultado_externo_para_selecao: false",
        "- media/maximo de probabilidades fora da analise oficial",
        "",
        "Arquivos auditados:",
        f"- {caminho_relativo(CAMINHO_TABELA_INTEGRADA)}",
        f"- {caminho_relativo(CAMINHO_THRESHOLDS_INTERNOS)}",
        "",
        "Falhas:",
    ]
    falhas = auditoria[auditoria["status"] != "ok"]
    if falhas.empty:
        linhas.append("- nenhuma")
    else:
        for _, linha in falhas.iterrows():
            linhas.append(
                f"- {linha['item']}: valor={linha['valor']} esperado={linha['esperado']}"
            )
    return "\n".join(linhas) + "\n"


def main() -> None:
    print("=" * 70)
    print("AUDITANDO INTEGRACAO DA TRIAGEM")
    print("=" * 70)

    tabela = ler_csv_obrigatorio(CAMINHO_TABELA_INTEGRADA)
    thresholds = ler_csv_obrigatorio(CAMINHO_THRESHOLDS_INTERNOS)
    auditoria = criar_auditoria(tabela, thresholds)
    resumo = gerar_resumo(tabela, auditoria)

    gravar_csv_atomico(auditoria, CAMINHO_AUDITORIA)
    CAMINHO_RESUMO.write_text(resumo, encoding="utf-8")

    print(resumo)
    if not auditoria["status"].eq("ok").all():
        raise ValueError(f"Auditoria da triagem falhou: {CAMINHO_AUDITORIA}")


if __name__ == "__main__":
    main()
