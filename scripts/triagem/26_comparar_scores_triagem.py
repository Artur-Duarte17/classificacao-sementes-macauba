from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 26 - COMPARAR SCORES PARA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Comparar formas alternativas de usar scores ja existentes
# - Ver se algum score cria baixo_risco seguro e util na validacao
# - Aplicar a melhor regra encontrada no teste
#
# Este script nao treina modelos e nao altera imagens.
# Ele deve ser executado manualmente pelo usuario.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TRIAGEM_TABELAS = PASTA_TABELAS / "07_triagem"
PASTA_TRIAGEM_LEGADA = PASTA_TABELAS / "07_fase2_triagem"

CAMINHO_PREDICOES = PASTA_TRIAGEM_TABELAS / "predicoes_todos_splits.csv"
CAMINHO_PREDICOES_LEGADO = PASTA_TRIAGEM_LEGADA / "predicoes_todos_splits_v2.csv"

CAMINHO_COMPARACAO_VALIDACAO = (
    PASTA_TRIAGEM_TABELAS / "comparacao_scores_validacao.csv"
)
CAMINHO_SCORE_RECOMENDADO = (
    PASTA_TRIAGEM_TABELAS / "score_triagem_recomendado.csv"
)
CAMINHO_AVALIACAO_TESTE = (
    PASTA_TRIAGEM_TABELAS / "avaliacao_score_recomendado_teste.csv"
)
CAMINHO_CASOS_CRITICOS = (
    PASTA_TRIAGEM_TABELAS / "casos_criticos_scores_triagem.csv"
)
CAMINHO_CONCLUSAO = (
    PASTA_TRIAGEM_TABELAS / "conclusao_comparacao_scores_triagem.txt"
)

THRESHOLD_BAIXO_INICIO = 0.05
THRESHOLD_BAIXO_FIM = 0.50
THRESHOLD_ALTO_INICIO = 0.50
THRESHOLD_ALTO_FIM = 0.95
PASSO_THRESHOLD = 0.01

MIN_NAO_CONTAMINADAS_BAIXO_RISCO = 5

ESTRATEGIAS = [
    {
        "estrategia_score": "prob_baseline",
        "tipo": "score_simples",
        "coluna_score": "prob_baseline_resnet18",
        "descricao": "usa apenas a probabilidade do baseline ResNet18",
    },
    {
        "estrategia_score": "prob_recortes",
        "tipo": "score_simples",
        "coluna_score": "prob_recortes_resnet18",
        "descricao": "usa apenas a probabilidade do modelo com recortes",
    },
    {
        "estrategia_score": "prob_media",
        "tipo": "score_simples",
        "coluna_score": "prob_media_modelos",
        "descricao": "usa a media entre baseline e recortes",
    },
    {
        "estrategia_score": "prob_max",
        "tipo": "score_simples",
        "coluna_score": "prob_max_modelos",
        "descricao": "usa a maior probabilidade entre baseline e recortes",
    },
    {
        "estrategia_score": "prob_min",
        "tipo": "score_simples",
        "coluna_score": "prob_min_modelos",
        "descricao": "usa a menor probabilidade entre baseline e recortes",
    },
    {
        "estrategia_score": "consenso_baixo_alerta_alto",
        "tipo": "consenso",
        "coluna_score": "",
        "descricao": (
            "baixo_risco somente se baseline e recortes forem baixos; "
            "alto_risco se baseline ou recortes forem altos"
        ),
    },
]


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho)


def resolver_entrada(caminho_atual: Path, caminho_legado: Path) -> Path:
    if caminho_atual.exists():
        return caminho_atual
    if caminho_legado.exists():
        print(f"AVISO: usando entrada legada: {caminho_legado}")
        return caminho_legado
    raise FileNotFoundError(
        f"Arquivo obrigatorio nao encontrado: {caminho_atual} nem {caminho_legado}"
    )


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


def gerar_faixa(inicio: float, fim: float, passo: float) -> list[float]:
    return [
        round(float(valor), 2)
        for valor in np.arange(inicio, fim + passo / 2, passo)
    ]


def preparar_predicoes(df: pd.DataFrame) -> pd.DataFrame:
    colunas_obrigatorias = [
        "split",
        "alvo",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "prob_media_modelos",
    ]
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {faltantes}")

    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["alvo"] = pd.to_numeric(df["alvo"], errors="coerce")

    for coluna in [
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "prob_media_modelos",
    ]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df[
        df["alvo"].isin([0, 1])
        & df["prob_baseline_resnet18"].notna()
        & df["prob_recortes_resnet18"].notna()
        & df["prob_media_modelos"].notna()
    ].copy()

    df["prob_max_modelos"] = df[
        ["prob_baseline_resnet18", "prob_recortes_resnet18"]
    ].max(axis=1)
    df["prob_min_modelos"] = df[
        ["prob_baseline_resnet18", "prob_recortes_resnet18"]
    ].min(axis=1)

    splits_necessarios = {"validacao", "teste"}
    splits_encontrados = set(df["split"].unique())
    faltantes = sorted(splits_necessarios - splits_encontrados)
    if faltantes:
        raise ValueError(f"Splits obrigatorios ausentes: {faltantes}")

    return df


def aplicar_regra(
    df: pd.DataFrame,
    estrategia: dict,
    regra_triagem: str,
    threshold_baixo,
    threshold_alto,
) -> pd.DataFrame:
    df = df.copy()
    tipo = estrategia["tipo"]

    if tipo == "score_simples":
        score = df[estrategia["coluna_score"]]

        if regra_triagem == "regra_3_zonas":
            condicoes = [
                score <= float(threshold_baixo),
                score >= float(threshold_alto),
            ]
            escolhas = ["baixo_risco", "alto_risco"]
            df["triagem_score"] = np.select(condicoes, escolhas, default="incerto")
        elif regra_triagem == "regra_2_zonas":
            df["triagem_score"] = np.where(
                score >= float(threshold_alto), "alto_risco", "incerto"
            )
        else:
            raise ValueError(f"Regra desconhecida: {regra_triagem}")

    elif tipo == "consenso":
        prob_baseline = df["prob_baseline_resnet18"]
        prob_recortes = df["prob_recortes_resnet18"]

        if regra_triagem == "regra_3_zonas":
            baixo = (
                (prob_baseline <= float(threshold_baixo))
                & (prob_recortes <= float(threshold_baixo))
            )
            alto = (
                (prob_baseline >= float(threshold_alto))
                | (prob_recortes >= float(threshold_alto))
            )
            df["triagem_score"] = np.select(
                [baixo, alto], ["baixo_risco", "alto_risco"], default="incerto"
            )
        elif regra_triagem == "regra_2_zonas":
            alto = (
                (prob_baseline >= float(threshold_alto))
                | (prob_recortes >= float(threshold_alto))
            )
            df["triagem_score"] = np.where(alto, "alto_risco", "incerto")
        else:
            raise ValueError(f"Regra desconhecida: {regra_triagem}")

    else:
        raise ValueError(f"Tipo de estrategia desconhecido: {tipo}")

    return df


def calcular_metricas(
    df: pd.DataFrame,
    split: str,
    estrategia: dict,
    regra_triagem: str,
    threshold_baixo,
    threshold_alto,
) -> dict:
    df = aplicar_regra(df, estrategia, regra_triagem, threshold_baixo, threshold_alto)

    total = len(df)
    total_contaminadas = int((df["alvo"] == 1).sum())
    total_nao_contaminadas = int((df["alvo"] == 0).sum())

    alto_risco = df["triagem_score"] == "alto_risco"
    baixo_risco = df["triagem_score"] == "baixo_risco"
    incerto = df["triagem_score"] == "incerto"
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

    baixo_risco_seguro = contaminadas_em_baixo_risco == 0
    baixo_risco_util = (
        baixo_risco_seguro
        and nao_contaminadas_em_baixo_risco >= MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    )

    return {
        "split": split,
        "estrategia_score": estrategia["estrategia_score"],
        "tipo_estrategia": estrategia["tipo"],
        "descricao_estrategia": estrategia["descricao"],
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
        "f1_alto_risco": f1_seguro(precisao_alto, cobertura_alto),
        "especificidade_operacional": dividir_seguro(
            nao_contaminadas_em_baixo_risco + nao_contaminadas_em_incerto,
            total_nao_contaminadas,
        ),
        "baixo_risco_seguro": baixo_risco_seguro,
        "baixo_risco_util": baixo_risco_util,
    }


def gerar_comparacao_validacao(df_validacao: pd.DataFrame) -> pd.DataFrame:
    thresholds_baixo = gerar_faixa(
        THRESHOLD_BAIXO_INICIO, THRESHOLD_BAIXO_FIM, PASSO_THRESHOLD
    )
    thresholds_alto = gerar_faixa(
        THRESHOLD_ALTO_INICIO, THRESHOLD_ALTO_FIM, PASSO_THRESHOLD
    )
    registros = []

    for estrategia in ESTRATEGIAS:
        for threshold_baixo in thresholds_baixo:
            for threshold_alto in thresholds_alto:
                if threshold_baixo >= threshold_alto:
                    continue
                registros.append(
                    calcular_metricas(
                        df_validacao,
                        split="validacao",
                        estrategia=estrategia,
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
                    estrategia=estrategia,
                    regra_triagem="regra_2_zonas",
                    threshold_baixo=pd.NA,
                    threshold_alto=threshold_alto,
                )
            )

    return pd.DataFrame(registros)


def escolher_regra_2_zonas(comparacao: pd.DataFrame) -> tuple[pd.Series, str]:
    candidatas = comparacao[comparacao["regra_triagem"] == "regra_2_zonas"].copy()
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
    return escolhida, "fallback_regra_2_zonas_sem_liberacao_automatica"


def escolher_score_recomendado(comparacao: pd.DataFrame) -> pd.DataFrame:
    tres_zonas = comparacao[comparacao["regra_triagem"] == "regra_3_zonas"].copy()
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
            "foi encontrado score com baixo_risco seguro e util na validacao: "
            "contaminadas_em_baixo_risco=0 e liberacao minima atendida"
        )
    else:
        escolhida, fallback = escolher_regra_2_zonas(comparacao)
        if len(seguras):
            melhor_por_score = (
                seguras.groupby("estrategia_score")["nao_contaminadas_em_baixo_risco"]
                .max()
                .sort_values(ascending=False)
            )
            melhor_texto = "; ".join(
                f"{score}={int(valor)}" for score, valor in melhor_por_score.items()
            )
            motivo = (
                "existem scores com baixo_risco seguro, mas nenhum atingiu "
                f"minimo de {MIN_NAO_CONTAMINADAS_BAIXO_RISCO} nao contaminadas. "
                f"Maximos seguros por score: {melhor_texto}. {fallback}"
            )
        else:
            motivo = (
                "nenhum score criou baixo_risco seguro na validacao. "
                f"{fallback}"
            )

    recomendada = escolhida.to_frame().T.copy()
    recomendada.insert(0, "origem_escolha", "validacao")
    recomendada["motivo_escolha"] = motivo
    recomendada["min_nao_contaminadas_baixo_risco"] = (
        MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    )
    return recomendada


def estrategia_por_nome(nome: str) -> dict:
    for estrategia in ESTRATEGIAS:
        if estrategia["estrategia_score"] == nome:
            return estrategia
    raise ValueError(f"Estrategia nao encontrada: {nome}")


def avaliar_recomendacao(
    df: pd.DataFrame,
    recomendada: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    linha = recomendada.iloc[0]
    estrategia = estrategia_por_nome(str(linha["estrategia_score"]))
    df_split = df[df["split"] == split].copy()

    metricas = calcular_metricas(
        df_split,
        split=split,
        estrategia=estrategia,
        regra_triagem=str(linha["regra_triagem"]),
        threshold_baixo=linha["threshold_baixo"],
        threshold_alto=float(linha["threshold_alto"]),
    )
    metricas["origem_thresholds"] = "validacao"
    metricas["motivo_escolha"] = linha["motivo_escolha"]

    return pd.DataFrame([metricas])


def gerar_casos_criticos(df: pd.DataFrame, recomendada: pd.DataFrame) -> pd.DataFrame:
    linha = recomendada.iloc[0]
    estrategia = estrategia_por_nome(str(linha["estrategia_score"]))
    df_avaliado = aplicar_regra(
        df[df["split"].isin(["validacao", "teste"])].copy(),
        estrategia=estrategia,
        regra_triagem=str(linha["regra_triagem"]),
        threshold_baixo=linha["threshold_baixo"],
        threshold_alto=float(linha["threshold_alto"]),
    )

    casos = df_avaliado[
        ((df_avaliado["alvo"] == 1) & (df_avaliado["triagem_score"] == "baixo_risco"))
        | ((df_avaliado["alvo"] == 0) & (df_avaliado["triagem_score"] == "alto_risco"))
    ].copy()

    if casos.empty:
        return pd.DataFrame(
            columns=[
                "tipo_caso",
                "split",
                "triagem_score",
                "alvo",
                "classe_real",
                "prob_media_modelos",
            ]
        )

    casos["tipo_caso"] = np.where(
        (casos["alvo"] == 1) & (casos["triagem_score"] == "baixo_risco"),
        "contaminada_em_baixo_risco",
        "nao_contaminada_em_alto_risco",
    )
    casos["estrategia_score"] = linha["estrategia_score"]
    casos["regra_triagem"] = linha["regra_triagem"]
    casos["threshold_baixo"] = linha["threshold_baixo"]
    casos["threshold_alto"] = linha["threshold_alto"]

    colunas_prioritarias = [
        "tipo_caso",
        "estrategia_score",
        "regra_triagem",
        "threshold_baixo",
        "threshold_alto",
        "split",
        "triagem_score",
        "alvo",
        "classe_real",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "prob_media_modelos",
        "prob_max_modelos",
        "prob_min_modelos",
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
    comparacao: pd.DataFrame,
    recomendada: pd.DataFrame,
    avaliacao_teste: pd.DataFrame,
) -> str:
    regra = recomendada.iloc[0]
    teste = avaliacao_teste.iloc[0]

    tres_zonas = comparacao[comparacao["regra_triagem"] == "regra_3_zonas"].copy()
    seguras = tres_zonas[tres_zonas["contaminadas_em_baixo_risco"] == 0].copy()
    seguras_uteis = seguras[
        seguras["nao_contaminadas_em_baixo_risco"]
        >= MIN_NAO_CONTAMINADAS_BAIXO_RISCO
    ].copy()

    resumo_scores = (
        seguras.groupby("estrategia_score")["nao_contaminadas_em_baixo_risco"]
        .max()
        .reset_index(name="max_nao_contaminadas_baixo_risco_seguro")
        .sort_values("max_nao_contaminadas_baixo_risco_seguro", ascending=False)
    )
    if resumo_scores.empty:
        resumo_texto = "nenhum score gerou baixo_risco seguro"
    else:
        resumo_texto = "; ".join(
            f"{linha.estrategia_score}: {int(linha.max_nao_contaminadas_baixo_risco_seguro)}"
            for linha in resumo_scores.itertuples(index=False)
        )

    linhas = [
        "COMPARACAO DE SCORES PARA TRIAGEM - FASE 2",
        "=" * 60,
        "",
        "Objetivo:",
        (
            "Testar se baseline, recortes, media, maximo, minimo ou consenso "
            "conseguem criar baixo_risco seguro e util na validacao."
        ),
        "",
        "Criterio cientifico:",
        (
            "baixo_risco so pode existir se contaminadas_em_baixo_risco = 0 "
            "na validacao."
        ),
        (
            "baixo_risco so e util se liberar pelo menos "
            f"{MIN_NAO_CONTAMINADAS_BAIXO_RISCO} sementes nao contaminadas."
        ),
        "",
        "Resumo da busca na validacao:",
        f"- regras de 3 zonas testadas: {len(tres_zonas)}",
        f"- regras de 3 zonas seguras: {len(seguras)}",
        f"- regras de 3 zonas seguras e uteis: {len(seguras_uteis)}",
        f"- maximos seguros por score: {resumo_texto}",
        "",
        "Regra recomendada:",
        f"- estrategia_score: {regra['estrategia_score']}",
        f"- regra_triagem: {regra['regra_triagem']}",
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

    if len(seguras_uteis):
        if int(teste["contaminadas_liberadas_por_engano"]) == 0:
            linhas.append(
                "Foi encontrada uma estrategia de score com baixo_risco seguro "
                "e util na validacao, e ela nao liberou contaminadas no teste. "
                "O baixo_risco pode ser considerado uma hipotese operacional "
                "preliminar para revisao tecnica."
            )
        else:
            linhas.append(
                "Uma estrategia passou na validacao, mas falhou no teste ao "
                "liberar contaminada em baixo_risco. Recomenda-se suspender "
                "baixo_risco e seguir com alto_risco/incerto."
            )
    else:
        linhas.append(
            "Nenhuma forma de score testada conseguiu baixo_risco seguro e util "
            "na validacao. A tentativa com scores RGB deve ser tratada como "
            "triagem conservadora, sem liberacao automatica."
        )

    return "\n".join(linhas) + "\n"


def main():
    print("=" * 60)
    print("COMPARANDO SCORES PARA TRIAGEM")
    print("=" * 60)

    PASTA_TRIAGEM_TABELAS.mkdir(parents=True, exist_ok=True)

    predicoes = preparar_predicoes(
        ler_csv_obrigatorio(resolver_entrada(CAMINHO_PREDICOES, CAMINHO_PREDICOES_LEGADO))
    )
    validacao = predicoes[predicoes["split"] == "validacao"].copy()

    comparacao = gerar_comparacao_validacao(validacao)
    recomendada = escolher_score_recomendado(comparacao)
    avaliacao_teste = avaliar_recomendacao(predicoes, recomendada, split="teste")
    casos_criticos = gerar_casos_criticos(predicoes, recomendada)
    conclusao = gerar_conclusao(comparacao, recomendada, avaliacao_teste)

    comparacao.to_csv(CAMINHO_COMPARACAO_VALIDACAO, index=False, encoding="utf-8-sig")
    recomendada.to_csv(CAMINHO_SCORE_RECOMENDADO, index=False, encoding="utf-8-sig")
    avaliacao_teste.to_csv(CAMINHO_AVALIACAO_TESTE, index=False, encoding="utf-8-sig")
    casos_criticos.to_csv(CAMINHO_CASOS_CRITICOS, index=False, encoding="utf-8-sig")
    CAMINHO_CONCLUSAO.write_text(conclusao, encoding="utf-8")

    print()
    print("Score/regra recomendado:")
    print(recomendada.to_string(index=False))
    print()
    print("Avaliacao no teste:")
    print(avaliacao_teste.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {CAMINHO_COMPARACAO_VALIDACAO}")
    print(f"- {CAMINHO_SCORE_RECOMENDADO}")
    print(f"- {CAMINHO_AVALIACAO_TESTE}")
    print(f"- {CAMINHO_CASOS_CRITICOS}")
    print(f"- {CAMINHO_CONCLUSAO}")
    print()
    print("Comparacao de scores da triagem concluida.")


if __name__ == "__main__":
    main()
