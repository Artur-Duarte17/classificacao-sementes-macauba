from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 25 - CALIBRAR THRESHOLDS DA TRIAGEM V2
# ------------------------------------------------------------
# Objetivo:
# - Usar validacao para escolher uma regra de triagem segura
# - Aplicar a regra escolhida no teste
# - Priorizar nao liberar sementes contaminadas
#
# Este script nao treina modelos e nao altera imagens.
# Ele deve ser executado manualmente pelo usuario.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_FASE2_TABELAS = PASTA_TABELAS / "07_fase2_triagem"

CAMINHO_PREDICOES = PASTA_FASE2_TABELAS / "predicoes_todos_splits_v2.csv"

CAMINHO_CALIBRACAO_VALIDACAO = (
    PASTA_FASE2_TABELAS / "calibracao_thresholds_validacao_v2.csv"
)
CAMINHO_THRESHOLDS_RECOMENDADOS = (
    PASTA_FASE2_TABELAS / "thresholds_triagem_recomendados_v2.csv"
)
CAMINHO_AVALIACAO_TESTE = (
    PASTA_FASE2_TABELAS / "avaliacao_triagem_calibrada_teste_v2.csv"
)
CAMINHO_CASOS_CRITICOS = (
    PASTA_FASE2_TABELAS / "casos_criticos_triagem_calibrada_v2.csv"
)
CAMINHO_CONCLUSAO = PASTA_FASE2_TABELAS / "conclusao_calibracao_triagem_v2.txt"

THRESHOLD_BAIXO_INICIO = 0.05
THRESHOLD_BAIXO_FIM = 0.50
THRESHOLD_ALTO_INICIO = 0.50
THRESHOLD_ALTO_FIM = 0.95
PASSO_THRESHOLD = 0.01

MIN_NAO_CONTAMINADAS_BAIXO_RISCO = 5


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho)


def dividir_seguro(numerador: float, denominador: float):
    if denominador == 0:
        return pd.NA
    return numerador / denominador


def f1_seguro(precisao, recall):
    if pd.isna(precisao) or pd.isna(recall):
        return pd.NA
    if precisao + recall == 0:
        return 0.0
    return 2 * precisao * recall / (precisao + recall)


def preparar_predicoes(df: pd.DataFrame) -> pd.DataFrame:
    colunas_obrigatorias = ["split", "alvo", "prob_media_modelos"]
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {faltantes}")

    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["alvo"] = pd.to_numeric(df["alvo"], errors="coerce")
    df["prob_media_modelos"] = pd.to_numeric(
        df["prob_media_modelos"], errors="coerce"
    )
    df = df[df["prob_media_modelos"].notna() & df["alvo"].isin([0, 1])].copy()

    splits_necessarios = {"validacao", "teste"}
    splits_encontrados = set(df["split"].unique())
    faltantes = sorted(splits_necessarios - splits_encontrados)
    if faltantes:
        raise ValueError(f"Splits obrigatorios ausentes: {faltantes}")

    return df


def gerar_faixa(inicio: float, fim: float, passo: float) -> list[float]:
    return [
        round(float(valor), 2)
        for valor in np.arange(inicio, fim + passo / 2, passo)
    ]


def classificar_regra_3_zonas(probabilidade, threshold_baixo, threshold_alto):
    if probabilidade <= threshold_baixo:
        return "baixo_risco"
    if probabilidade >= threshold_alto:
        return "alto_risco"
    return "incerto"


def classificar_regra_2_zonas(probabilidade, threshold_alto):
    if probabilidade >= threshold_alto:
        return "alto_risco"
    return "incerto"


def aplicar_regra(df: pd.DataFrame, regra_triagem: str, threshold_baixo, threshold_alto):
    df = df.copy()

    if regra_triagem == "regra_3_zonas":
        df["triagem_calibrada"] = df["prob_media_modelos"].map(
            lambda prob: classificar_regra_3_zonas(
                prob, threshold_baixo, threshold_alto
            )
        )
    elif regra_triagem == "regra_2_zonas":
        df["triagem_calibrada"] = df["prob_media_modelos"].map(
            lambda prob: classificar_regra_2_zonas(prob, threshold_alto)
        )
    else:
        raise ValueError(f"Regra desconhecida: {regra_triagem}")

    return df


def calcular_metricas(
    df: pd.DataFrame,
    split: str,
    regra_triagem: str,
    threshold_baixo,
    threshold_alto,
) -> dict:
    df = aplicar_regra(df, regra_triagem, threshold_baixo, threshold_alto)

    total = len(df)
    total_contaminadas = int((df["alvo"] == 1).sum())
    total_nao_contaminadas = int((df["alvo"] == 0).sum())

    alto_risco = df["triagem_calibrada"] == "alto_risco"
    baixo_risco = df["triagem_calibrada"] == "baixo_risco"
    incerto = df["triagem_calibrada"] == "incerto"
    contaminada = df["alvo"] == 1
    nao_contaminada = df["alvo"] == 0

    sementes_alto_risco = int(alto_risco.sum())
    sementes_baixo_risco = int(baixo_risco.sum())
    sementes_incerto = int(incerto.sum())
    contaminadas_em_alto_risco = int((alto_risco & contaminada).sum())
    contaminadas_em_baixo_risco = int((baixo_risco & contaminada).sum())
    contaminadas_em_incerto = int((incerto & contaminada).sum())
    nao_contaminadas_em_alto_risco = int((alto_risco & nao_contaminada).sum())
    nao_contaminadas_em_baixo_risco = int((baixo_risco & nao_contaminada).sum())
    nao_contaminadas_em_incerto = int((incerto & nao_contaminada).sum())

    precisao_alto = dividir_seguro(contaminadas_em_alto_risco, sementes_alto_risco)
    cobertura_alto = dividir_seguro(contaminadas_em_alto_risco, total_contaminadas)
    f1_alto = f1_seguro(precisao_alto, cobertura_alto)

    baixo_risco_seguro = contaminadas_em_baixo_risco == 0
    baixo_risco_util = (
        baixo_risco_seguro
        and nao_contaminadas_em_baixo_risco >= MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    )

    return {
        "split": split,
        "regra_triagem": regra_triagem,
        "threshold_baixo": threshold_baixo,
        "threshold_alto": threshold_alto,
        "total": total,
        "total_contaminadas": total_contaminadas,
        "total_nao_contaminadas": total_nao_contaminadas,
        "sementes_alto_risco": sementes_alto_risco,
        "sementes_baixo_risco": sementes_baixo_risco,
        "sementes_incerto": sementes_incerto,
        "contaminadas_em_alto_risco": contaminadas_em_alto_risco,
        "contaminadas_em_baixo_risco": contaminadas_em_baixo_risco,
        "contaminadas_em_incerto": contaminadas_em_incerto,
        "nao_contaminadas_em_alto_risco": nao_contaminadas_em_alto_risco,
        "nao_contaminadas_em_baixo_risco": nao_contaminadas_em_baixo_risco,
        "nao_contaminadas_em_incerto": nao_contaminadas_em_incerto,
        "contaminadas_liberadas_por_engano": contaminadas_em_baixo_risco,
        "nao_contaminadas_separadas_por_cautela": nao_contaminadas_em_alto_risco,
        "taxa_revisao": dividir_seguro(sementes_incerto, total),
        "taxa_liberacao": dividir_seguro(sementes_baixo_risco, total),
        "risco_da_liberacao": dividir_seguro(
            contaminadas_em_baixo_risco, sementes_baixo_risco
        ),
        "precisao_alto_risco": precisao_alto,
        "cobertura_alto_risco": cobertura_alto,
        "f1_alto_risco": f1_alto,
        "especificidade_operacional": dividir_seguro(
            nao_contaminadas_em_baixo_risco + nao_contaminadas_em_incerto,
            total_nao_contaminadas,
        ),
        "baixo_risco_seguro": baixo_risco_seguro,
        "baixo_risco_util": baixo_risco_util,
    }


def gerar_calibracao_validacao(df_validacao: pd.DataFrame) -> pd.DataFrame:
    thresholds_baixo = gerar_faixa(
        THRESHOLD_BAIXO_INICIO, THRESHOLD_BAIXO_FIM, PASSO_THRESHOLD
    )
    thresholds_alto = gerar_faixa(
        THRESHOLD_ALTO_INICIO, THRESHOLD_ALTO_FIM, PASSO_THRESHOLD
    )
    registros = []

    for threshold_baixo in thresholds_baixo:
        for threshold_alto in thresholds_alto:
            if threshold_baixo >= threshold_alto:
                continue
            registros.append(
                calcular_metricas(
                    df_validacao,
                    split="validacao",
                    regra_triagem="regra_3_zonas",
                    threshold_baixo=threshold_baixo,
                    threshold_alto=threshold_alto,
                )
            )

    for threshold_alto in thresholds_alto:
        registros.append(
            calcular_metricas(
                df_validacao,
                split="validacao",
                regra_triagem="regra_2_zonas",
                threshold_baixo=pd.NA,
                threshold_alto=threshold_alto,
            )
        )

    return pd.DataFrame(registros)


def escolher_regra_2_zonas(calibracao: pd.DataFrame) -> tuple[pd.Series, str]:
    candidatas = calibracao[calibracao["regra_triagem"] == "regra_2_zonas"].copy()
    escolhida = candidatas.sort_values(
        [
            "f1_alto_risco",
            "cobertura_alto_risco",
            "precisao_alto_risco",
            "nao_contaminadas_em_alto_risco",
            "taxa_revisao",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    return escolhida, "regra_2_zonas_sem_liberacao_automatica"


def escolher_regra_recomendada(calibracao: pd.DataFrame) -> pd.DataFrame:
    tres_zonas = calibracao[calibracao["regra_triagem"] == "regra_3_zonas"].copy()
    seguras = tres_zonas[tres_zonas["contaminadas_em_baixo_risco"] == 0].copy()
    seguras_uteis = seguras[
        seguras["nao_contaminadas_em_baixo_risco"]
        >= MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    ].copy()

    if len(seguras_uteis):
        escolhida = seguras_uteis.sort_values(
            [
                "nao_contaminadas_em_baixo_risco",
                "cobertura_alto_risco",
                "precisao_alto_risco",
                "nao_contaminadas_em_alto_risco",
                "taxa_revisao",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        motivo = (
            "regra_3_zonas_segura_e_util_na_validacao: "
            "contaminadas_em_baixo_risco=0 e baixo_risco atingiu "
            f"minimo de {MIN_NAO_CONTAMINADAS_BAIXO_RISCO} nao contaminadas"
        )
    else:
        escolhida, complemento = escolher_regra_2_zonas(calibracao)
        if len(seguras):
            max_liberacao_segura = int(seguras["nao_contaminadas_em_baixo_risco"].max())
            motivo = (
                "existe_regra_3_zonas_segura_na_validacao, mas baixo_risco "
                "nao atingiu utilidade minima. Maximo seguro de nao "
                f"contaminadas liberadas: {max_liberacao_segura}. "
                f"Fallback: {complemento}"
            )
        else:
            motivo = (
                "nenhuma_regra_3_zonas_segura_na_validacao. "
                f"Fallback: {complemento}"
            )

    recomendada = escolhida.to_frame().T.copy()
    recomendada.insert(0, "origem_escolha", "validacao")
    recomendada["motivo_escolha"] = motivo
    recomendada["min_nao_contaminadas_baixo_risco"] = (
        MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    )
    return recomendada


def avaliar_regra_recomendada(
    df: pd.DataFrame,
    recomendada: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    linha = recomendada.iloc[0]
    df_split = df[df["split"] == split].copy()

    metricas = calcular_metricas(
        df_split,
        split=split,
        regra_triagem=str(linha["regra_triagem"]),
        threshold_baixo=linha["threshold_baixo"],
        threshold_alto=float(linha["threshold_alto"]),
    )
    metricas["origem_thresholds"] = "validacao"
    metricas["motivo_escolha"] = linha["motivo_escolha"]

    return pd.DataFrame([metricas])


def gerar_casos_criticos(df: pd.DataFrame, recomendada: pd.DataFrame) -> pd.DataFrame:
    linha = recomendada.iloc[0]
    df_avaliado = aplicar_regra(
        df[df["split"].isin(["validacao", "teste"])].copy(),
        regra_triagem=str(linha["regra_triagem"]),
        threshold_baixo=linha["threshold_baixo"],
        threshold_alto=float(linha["threshold_alto"]),
    )

    casos = df_avaliado[
        ((df_avaliado["alvo"] == 1) & (df_avaliado["triagem_calibrada"] == "baixo_risco"))
        | ((df_avaliado["alvo"] == 0) & (df_avaliado["triagem_calibrada"] == "alto_risco"))
    ].copy()

    if casos.empty:
        return pd.DataFrame(
            columns=[
                "tipo_caso",
                "split",
                "triagem_calibrada",
                "alvo",
                "classe_real",
                "prob_media_modelos",
            ]
        )

    casos["tipo_caso"] = np.where(
        (casos["alvo"] == 1) & (casos["triagem_calibrada"] == "baixo_risco"),
        "contaminada_em_baixo_risco",
        "nao_contaminada_em_alto_risco",
    )

    colunas_prioritarias = [
        "tipo_caso",
        "split",
        "triagem_calibrada",
        "alvo",
        "classe_real",
        "prob_media_modelos",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "nome_arquivo",
        "caminho_imagem_original",
        "caminho_imagem_baseline",
        "caminho_imagem_recortes",
    ]
    colunas_existentes = [coluna for coluna in colunas_prioritarias if coluna in casos.columns]
    colunas_restantes = [coluna for coluna in casos.columns if coluna not in colunas_existentes]

    return casos[colunas_existentes + colunas_restantes].sort_values(
        ["tipo_caso", "split", "prob_media_modelos"]
    )


def formatar_percentual(valor) -> str:
    if pd.isna(valor):
        return "n/a"
    return f"{float(valor) * 100:.2f}%"


def gerar_conclusao(
    recomendada: pd.DataFrame,
    avaliacao_teste: pd.DataFrame,
    calibracao: pd.DataFrame,
) -> str:
    regra = recomendada.iloc[0]
    teste = avaliacao_teste.iloc[0]
    regra_nome = str(regra["regra_triagem"])
    seguras = calibracao[
        (calibracao["regra_triagem"] == "regra_3_zonas")
        & (calibracao["contaminadas_em_baixo_risco"] == 0)
    ]
    seguras_uteis = seguras[
        seguras["nao_contaminadas_em_baixo_risco"]
        >= MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    ]

    linhas = [
        "CALIBRACAO OPERACIONAL DA TRIAGEM - FASE 2",
        "=" * 60,
        "",
        "Regra cientifica principal:",
        (
            "baixo_risco so pode existir se contaminadas_em_baixo_risco = 0 "
            "na validacao."
        ),
        (
            "baixo_risco so e operacionalmente util se liberar pelo menos "
            f"{MIN_NAO_CONTAMINADAS_BAIXO_RISCO} sementes nao contaminadas."
        ),
        "",
        "Busca realizada:",
        (
            f"- threshold_baixo: {THRESHOLD_BAIXO_INICIO:.2f} a "
            f"{THRESHOLD_BAIXO_FIM:.2f}, passo {PASSO_THRESHOLD:.2f}"
        ),
        (
            f"- threshold_alto: {THRESHOLD_ALTO_INICIO:.2f} a "
            f"{THRESHOLD_ALTO_FIM:.2f}, passo {PASSO_THRESHOLD:.2f}"
        ),
        f"- regras de 3 zonas seguras na validacao: {len(seguras)}",
        f"- regras de 3 zonas seguras e uteis na validacao: {len(seguras_uteis)}",
        "",
        "Regra recomendada pela validacao:",
        f"- regra: {regra_nome}",
        f"- threshold_baixo: {regra['threshold_baixo']}",
        f"- threshold_alto: {float(regra['threshold_alto']):.2f}",
        f"- motivo: {regra['motivo_escolha']}",
        "",
        "Aplicacao no teste:",
        f"- total: {int(teste['total'])}",
        f"- alto_risco: {int(teste['sementes_alto_risco'])}",
        f"- baixo_risco: {int(teste['sementes_baixo_risco'])}",
        f"- incerto: {int(teste['sementes_incerto'])}",
        (
            "- contaminadas liberadas por engano: "
            f"{int(teste['contaminadas_liberadas_por_engano'])}"
        ),
        f"- taxa de revisao: {formatar_percentual(teste['taxa_revisao'])}",
        f"- cobertura alto_risco: {formatar_percentual(teste['cobertura_alto_risco'])}",
        f"- precisao alto_risco: {formatar_percentual(teste['precisao_alto_risco'])}",
        "",
        "Conclusao operacional:",
    ]

    if regra_nome == "regra_3_zonas":
        if int(teste["contaminadas_liberadas_por_engano"]) == 0:
            linhas.append(
                "Foi encontrada uma regra de 3 zonas segura e util na validacao, "
                "e ela nao liberou contaminadas no teste. O baixo_risco pode ser "
                "considerado uma opcao operacional preliminar, mantendo revisao "
                "continua dos casos criticos."
            )
        else:
            linhas.append(
                "A regra de 3 zonas foi segura na validacao, mas falhou no teste "
                "ao liberar contaminada em baixo_risco. Recomenda-se suspender "
                "baixo_risco e usar temporariamente alto_risco e incerto/revisao "
                "manual."
            )
    else:
        linhas.append(
            "Nao foi encontrada regra de 3 zonas segura e util para liberacao "
            "automatica de baixo_risco na validacao. Recomenda-se usar "
            "temporariamente apenas alto_risco e incerto/revisao manual."
        )

    return "\n".join(linhas) + "\n"


def main():
    print("=" * 60)
    print("CALIBRANDO THRESHOLDS DA TRIAGEM V2")
    print("=" * 60)

    PASTA_FASE2_TABELAS.mkdir(parents=True, exist_ok=True)

    predicoes = preparar_predicoes(ler_csv_obrigatorio(CAMINHO_PREDICOES))
    validacao = predicoes[predicoes["split"] == "validacao"].copy()

    calibracao = gerar_calibracao_validacao(validacao)
    recomendada = escolher_regra_recomendada(calibracao)
    avaliacao_teste = avaliar_regra_recomendada(
        predicoes, recomendada, split="teste"
    )
    casos_criticos = gerar_casos_criticos(predicoes, recomendada)
    conclusao = gerar_conclusao(recomendada, avaliacao_teste, calibracao)

    calibracao.to_csv(
        CAMINHO_CALIBRACAO_VALIDACAO, index=False, encoding="utf-8-sig"
    )
    recomendada.to_csv(
        CAMINHO_THRESHOLDS_RECOMENDADOS, index=False, encoding="utf-8-sig"
    )
    avaliacao_teste.to_csv(CAMINHO_AVALIACAO_TESTE, index=False, encoding="utf-8-sig")
    casos_criticos.to_csv(CAMINHO_CASOS_CRITICOS, index=False, encoding="utf-8-sig")
    CAMINHO_CONCLUSAO.write_text(conclusao, encoding="utf-8")

    print()
    print("Regra recomendada:")
    print(recomendada.to_string(index=False))
    print()
    print("Avaliacao no teste:")
    print(avaliacao_teste.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {CAMINHO_CALIBRACAO_VALIDACAO}")
    print(f"- {CAMINHO_THRESHOLDS_RECOMENDADOS}")
    print(f"- {CAMINHO_AVALIACAO_TESTE}")
    print(f"- {CAMINHO_CASOS_CRITICOS}")
    print(f"- {CAMINHO_CONCLUSAO}")
    print()
    print("Calibracao da triagem v2 concluida.")


if __name__ == "__main__":
    main()
