from pathlib import Path

import pandas as pd


# ============================================================
# SCRIPT 31 - ANALISAR TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Avaliar a triagem como uma operacao real
# - Priorizar seguranca: nao liberar sementes contaminadas
# - Comparar regra de 3 zonas contra regra conservadora de 2 zonas
#
# Esta analise e preliminar, pois os thresholds ainda nao foram
# calibrados com predicoes do conjunto de validacao.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TRIAGEM_TABELAS = PASTA_TABELAS / "08_triagem"
PASTA_TRIAGEM_LEGADA = PASTA_TABELAS / "07_triagem"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM_TABELAS / "tabela_integrada.csv"
CAMINHO_TABELA_INTEGRADA_LEGADA = PASTA_TRIAGEM_LEGADA / "tabela_mestre_v2.csv"

LIMIAR_ALTO_RISCO = 0.70
LIMIAR_BAIXO_RISCO = 0.30

TRIAGENS_COM_PREDICAO = ["alto_risco", "baixo_risco", "incerto"]
REGRAS_TRIAGEM = ["regra_3_zonas", "regra_2_zonas"]
ORDEM_TRIAGENS = ["alto_risco", "baixo_risco", "incerto", "sem_predicao"]


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho)


def resolver_entrada(caminho_atual: Path, caminho_legado: Path) -> Path:
    if caminho_atual.exists():
        return caminho_atual
    if caminho_legado.exists():
        print(f"AVISO: usando entrada anterior: {caminho_legado}")
        return caminho_legado
    raise FileNotFoundError(
        f"Arquivo obrigatorio nao encontrado: {caminho_atual} nem {caminho_legado}"
    )


def classificar_regra_3_zonas(probabilidade):
    if pd.isna(probabilidade):
        return "sem_predicao"
    if probabilidade >= LIMIAR_ALTO_RISCO:
        return "alto_risco"
    if probabilidade <= LIMIAR_BAIXO_RISCO:
        return "baixo_risco"
    return "incerto"


def classificar_regra_2_zonas(probabilidade):
    if pd.isna(probabilidade):
        return "sem_predicao"
    if probabilidade >= LIMIAR_ALTO_RISCO:
        return "alto_risco"
    return "incerto"


def dividir_seguro(numerador: float, denominador: float):
    if denominador == 0:
        return pd.NA
    return numerador / denominador


def preparar_tabela_com_predicao(tabela: pd.DataFrame) -> pd.DataFrame:
    colunas_obrigatorias = ["contaminou", "prob_media_modelos"]
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in tabela.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {faltantes}")

    tabela = tabela.copy()
    tabela["contaminou"] = pd.to_numeric(tabela["contaminou"], errors="coerce")
    tabela["prob_media_modelos"] = pd.to_numeric(
        tabela["prob_media_modelos"], errors="coerce"
    )

    return tabela[tabela["prob_media_modelos"].notna()].copy()


def simular_regras(tabela_com_predicao: pd.DataFrame) -> pd.DataFrame:
    simulacoes = []

    for regra_triagem, classificador in [
        ("regra_3_zonas", classificar_regra_3_zonas),
        ("regra_2_zonas", classificar_regra_2_zonas),
    ]:
        simulacao = tabela_com_predicao.copy()
        simulacao["regra_triagem"] = regra_triagem
        simulacao["triagem"] = simulacao["prob_media_modelos"].map(classificador)
        simulacoes.append(simulacao)

    return pd.concat(simulacoes, ignore_index=True)


def gerar_contagens(simulacao: pd.DataFrame) -> pd.DataFrame:
    contagens = (
        simulacao.groupby(["regra_triagem", "triagem", "contaminou"], dropna=False)
        .size()
        .reset_index(name="quantidade")
    )
    contagens["regra_triagem"] = pd.Categorical(
        contagens["regra_triagem"], categories=REGRAS_TRIAGEM, ordered=True
    )
    contagens["triagem"] = pd.Categorical(
        contagens["triagem"], categories=ORDEM_TRIAGENS, ordered=True
    )
    contagens = contagens.sort_values(["regra_triagem", "triagem", "contaminou"])
    contagens["regra_triagem"] = contagens["regra_triagem"].astype(str)
    contagens["triagem"] = contagens["triagem"].astype(str)
    return contagens


def calcular_metricas_regra(df_regra: pd.DataFrame, regra_triagem: str) -> dict:
    total = len(df_regra)
    total_contaminadas = int((df_regra["contaminou"] == 1).sum())
    total_nao_contaminadas = int((df_regra["contaminou"] == 0).sum())

    alto_risco = df_regra["triagem"] == "alto_risco"
    baixo_risco = df_regra["triagem"] == "baixo_risco"
    incerto = df_regra["triagem"] == "incerto"
    contaminada = df_regra["contaminou"] == 1
    nao_contaminada = df_regra["contaminou"] == 0

    separadas_automaticamente = int(alto_risco.sum())
    liberadas_automaticamente = int(baixo_risco.sum())
    revisao_manual = int(incerto.sum())
    contaminadas_em_alto_risco = int((alto_risco & contaminada).sum())
    contaminadas_em_baixo_risco = int((baixo_risco & contaminada).sum())
    nao_contaminadas_em_alto_risco = int((alto_risco & nao_contaminada).sum())
    nao_contaminadas_em_baixo_risco = int((baixo_risco & nao_contaminada).sum())
    contaminadas_em_incerto = int((incerto & contaminada).sum())
    nao_contaminadas_em_incerto = int((incerto & nao_contaminada).sum())

    baixo_risco_seguro = contaminadas_em_baixo_risco == 0
    recomendacao_baixo_risco = (
        "baixo_risco_seguro_preliminar"
        if baixo_risco_seguro
        else "suspender_baixo_risco"
    )

    return {
        "regra_triagem": regra_triagem,
        "total_com_predicao": total,
        "total_contaminadas": total_contaminadas,
        "total_nao_contaminadas": total_nao_contaminadas,
        "sementes_separadas_automaticamente": separadas_automaticamente,
        "sementes_liberadas_automaticamente": liberadas_automaticamente,
        "sementes_revisao_manual": revisao_manual,
        "contaminadas_em_alto_risco": contaminadas_em_alto_risco,
        "contaminadas_em_baixo_risco": contaminadas_em_baixo_risco,
        "contaminadas_em_incerto": contaminadas_em_incerto,
        "nao_contaminadas_em_alto_risco": nao_contaminadas_em_alto_risco,
        "nao_contaminadas_em_baixo_risco": nao_contaminadas_em_baixo_risco,
        "nao_contaminadas_em_incerto": nao_contaminadas_em_incerto,
        "contaminadas_liberadas_por_engano": contaminadas_em_baixo_risco,
        "nao_contaminadas_separadas_por_cautela": nao_contaminadas_em_alto_risco,
        "taxa_revisao": dividir_seguro(revisao_manual, total),
        "precisao_alto_risco": dividir_seguro(
            contaminadas_em_alto_risco, separadas_automaticamente
        ),
        "cobertura_alto_risco": dividir_seguro(
            contaminadas_em_alto_risco, total_contaminadas
        ),
        "taxa_liberacao": dividir_seguro(liberadas_automaticamente, total),
        "risco_da_liberacao": dividir_seguro(
            contaminadas_em_baixo_risco, liberadas_automaticamente
        ),
        "especificidade_operacional": dividir_seguro(
            nao_contaminadas_em_baixo_risco + nao_contaminadas_em_incerto,
            total_nao_contaminadas,
        ),
        "baixo_risco_seguro": baixo_risco_seguro,
        "recomendacao_baixo_risco": recomendacao_baixo_risco,
    }


def calcular_metricas(simulacao: pd.DataFrame) -> pd.DataFrame:
    metricas = [
        calcular_metricas_regra(df_regra, regra_triagem)
        for regra_triagem, df_regra in simulacao.groupby("regra_triagem", sort=False)
    ]
    return pd.DataFrame(metricas)


def gerar_analise_agregada(
    simulacao: pd.DataFrame, coluna: str, nome_coluna_saida: str
) -> pd.DataFrame:
    if coluna not in simulacao.columns:
        return pd.DataFrame(
            columns=[
                "regra_triagem",
                nome_coluna_saida,
                "triagem",
                "contaminou",
                "quantidade",
            ]
        )

    analise = (
        simulacao.groupby(["regra_triagem", coluna, "triagem", "contaminou"], dropna=False)
        .size()
        .reset_index(name="quantidade")
        .rename(columns={coluna: nome_coluna_saida})
    )
    analise["regra_triagem"] = pd.Categorical(
        analise["regra_triagem"], categories=REGRAS_TRIAGEM, ordered=True
    )
    analise["triagem"] = pd.Categorical(
        analise["triagem"], categories=ORDEM_TRIAGENS, ordered=True
    )
    analise = analise.sort_values(
        ["regra_triagem", nome_coluna_saida, "triagem", "contaminou"]
    )
    analise["regra_triagem"] = analise["regra_triagem"].astype(str)
    analise["triagem"] = analise["triagem"].astype(str)
    return analise


def selecionar_casos_baixo_risco_contaminados(simulacao: pd.DataFrame) -> pd.DataFrame:
    casos = simulacao[
        (simulacao["regra_triagem"] == "regra_3_zonas")
        & (simulacao["triagem"] == "baixo_risco")
        & (simulacao["contaminou"] == 1)
    ].copy()

    colunas_prioritarias = [
        "regra_triagem",
        "triagem",
        "contaminou",
        "prob_media_modelos",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "classe",
        "experimento_rotulo",
        "tratamento_planilha",
        "pasta_esperada",
        "origem_planilha",
        "id_semente_original",
        "caminho_relativo",
        "nome_copiado",
        "split",
    ]
    colunas_existentes = [coluna for coluna in colunas_prioritarias if coluna in casos.columns]
    colunas_restantes = [coluna for coluna in casos.columns if coluna not in colunas_existentes]

    return casos[colunas_existentes + colunas_restantes]


def formatar_percentual(valor) -> str:
    if pd.isna(valor):
        return "n/a"
    return f"{float(valor) * 100:.2f}%"


def formatar_quantidade(valor: int, singular: str, plural: str) -> str:
    palavra = singular if valor == 1 else plural
    return f"{valor} {palavra}"


def obter_linha_metrica(metricas: pd.DataFrame, regra_triagem: str) -> pd.Series:
    linha = metricas[metricas["regra_triagem"] == regra_triagem]
    if linha.empty:
        raise ValueError(f"Metricas nao encontradas para {regra_triagem}")
    return linha.iloc[0]


def gerar_conclusao(
    tabela: pd.DataFrame,
    tabela_com_predicao: pd.DataFrame,
    metricas: pd.DataFrame,
    casos_baixo_risco_contaminados: pd.DataFrame,
) -> str:
    sem_predicao = int(tabela["prob_media_modelos"].isna().sum())
    total_original = len(tabela)
    total_com_predicao = len(tabela_com_predicao)
    regra_3 = obter_linha_metrica(metricas, "regra_3_zonas")
    regra_2 = obter_linha_metrica(metricas, "regra_2_zonas")
    contaminadas_baixo_risco = int(regra_3["contaminadas_em_baixo_risco"])

    linhas = [
        "ANALISE OPERACIONAL CONSERVADORA DA TRIAGEM",
        "=" * 60,
        "",
        "Status da analise:",
        (
            "Esta analise e preliminar, pois os thresholds ainda nao foram "
            "calibrados com predicoes do conjunto de validacao."
        ),
        "",
        "Base analisada:",
        f"- Total de registros na tabela integrada: {total_original}",
        f"- Registros com predicao analisados: {total_com_predicao}",
        f"- Registros sem predicao fora da analise principal: {sem_predicao}",
        "",
        "Regra de 3 zonas:",
        f"- Separadas automaticamente como alto_risco: {int(regra_3['sementes_separadas_automaticamente'])}",
        f"- Liberadas automaticamente como baixo_risco: {int(regra_3['sementes_liberadas_automaticamente'])}",
        f"- Enviadas para revisao manual como incerto: {int(regra_3['sementes_revisao_manual'])}",
        f"- Contaminadas liberadas por engano: {contaminadas_baixo_risco}",
        f"- Nao contaminadas separadas por cautela: {int(regra_3['nao_contaminadas_separadas_por_cautela'])}",
        f"- Taxa de revisao: {formatar_percentual(regra_3['taxa_revisao'])}",
        f"- Risco da liberacao: {formatar_percentual(regra_3['risco_da_liberacao'])}",
        "",
        "Regra de 2 zonas:",
        f"- Separadas automaticamente como alto_risco: {int(regra_2['sementes_separadas_automaticamente'])}",
        f"- Liberadas automaticamente como baixo_risco: {int(regra_2['sementes_liberadas_automaticamente'])}",
        f"- Enviadas para revisao manual como incerto: {int(regra_2['sementes_revisao_manual'])}",
        f"- Contaminadas liberadas por engano: {int(regra_2['contaminadas_liberadas_por_engano'])}",
        f"- Taxa de revisao: {formatar_percentual(regra_2['taxa_revisao'])}",
        "",
        "Criterio de seguranca:",
        (
            "baixo_risco e considerado seguro somente se "
            "contaminadas_em_baixo_risco = 0."
        ),
    ]

    if contaminadas_baixo_risco > 0:
        linhas.extend(
            [
                (
                    "Foram encontradas "
                    f"{formatar_quantidade(contaminadas_baixo_risco, 'semente contaminada', 'sementes contaminadas')} "
                    "em baixo_risco."
                ),
                (
                    f"Arquivo com os casos: {PASTA_TRIAGEM_TABELAS / 'casos_baixo_risco_contaminados.csv'}"
                ),
                "",
                "Conclusao operacional preliminar:",
                (
                    "A regra de 3 zonas nao e recomendada neste momento, pois "
                    "houve semente contaminada classificada como baixo_risco. "
                    "Por seguranca, recomenda-se suspender a liberacao "
                    "automatica e utilizar temporariamente a regra de 2 zonas: "
                    "alto_risco e incerto/revisao manual. A calibracao "
                    "definitiva deve ser feita posteriormente usando predicoes "
                    "do conjunto de validacao."
                ),
            ]
        )
    else:
        linhas.extend(
            [
                "Nao foram encontradas sementes contaminadas em baixo_risco.",
                "",
                "Conclusao operacional preliminar:",
                (
                    "A regra de 3 zonas nao liberou sementes contaminadas nesta "
                    "analise preliminar. O baixo_risco pode ser tratado como "
                    "seguro apenas provisoriamente, ate calibracao definitiva "
                    "com predicoes do conjunto de validacao."
                ),
            ]
        )

    if casos_baixo_risco_contaminados.empty:
        linhas.append("")
        linhas.append("Nenhum caso contaminado em baixo_risco foi listado.")

    return "\n".join(linhas) + "\n"


def main():
    print("=" * 60)
    print("ANALISANDO TRIAGEM")
    print("=" * 60)

    PASTA_TRIAGEM_TABELAS.mkdir(parents=True, exist_ok=True)

    tabela = ler_csv_obrigatorio(
        resolver_entrada(CAMINHO_TABELA_INTEGRADA, CAMINHO_TABELA_INTEGRADA_LEGADA)
    )
    tabela_com_predicao = preparar_tabela_com_predicao(tabela)
    simulacao = simular_regras(tabela_com_predicao)

    analise_triagem = gerar_contagens(simulacao)
    metricas_triagem = calcular_metricas(simulacao)
    triagem_por_tratamento = gerar_analise_agregada(
        simulacao, "tratamento_planilha", "tratamento_planilha"
    )
    triagem_por_origem = gerar_analise_agregada(
        simulacao, "origem_planilha", "origem_planilha"
    )
    casos_baixo_risco_contaminados = selecionar_casos_baixo_risco_contaminados(
        simulacao
    )
    conclusao = gerar_conclusao(
        tabela,
        tabela_com_predicao,
        metricas_triagem,
        casos_baixo_risco_contaminados,
    )

    caminhos_saida = {
        "analise_triagem": PASTA_TRIAGEM_TABELAS / "analise_triagem.csv",
        "metricas_triagem": PASTA_TRIAGEM_TABELAS / "metricas_triagem.csv",
        "triagem_por_tratamento": PASTA_TRIAGEM_TABELAS
        / "triagem_por_tratamento.csv",
        "triagem_por_origem": PASTA_TRIAGEM_TABELAS / "triagem_por_origem.csv",
        "casos_baixo_risco_contaminados": PASTA_TRIAGEM_TABELAS
        / "casos_baixo_risco_contaminados.csv",
        "simulacao_regras": PASTA_TRIAGEM_TABELAS / "simulacao_regras_triagem.csv",
        "conclusao": PASTA_TRIAGEM_TABELAS / "conclusao_triagem.txt",
    }

    analise_triagem.to_csv(
        caminhos_saida["analise_triagem"], index=False, encoding="utf-8-sig"
    )
    metricas_triagem.to_csv(
        caminhos_saida["metricas_triagem"], index=False, encoding="utf-8-sig"
    )
    triagem_por_tratamento.to_csv(
        caminhos_saida["triagem_por_tratamento"], index=False, encoding="utf-8-sig"
    )
    triagem_por_origem.to_csv(
        caminhos_saida["triagem_por_origem"], index=False, encoding="utf-8-sig"
    )
    casos_baixo_risco_contaminados.to_csv(
        caminhos_saida["casos_baixo_risco_contaminados"],
        index=False,
        encoding="utf-8-sig",
    )
    simulacao.to_csv(
        caminhos_saida["simulacao_regras"], index=False, encoding="utf-8-sig"
    )
    caminhos_saida["conclusao"].write_text(conclusao, encoding="utf-8")

    print()
    print("Metricas operacionais:")
    print(metricas_triagem.to_string(index=False))
    print()
    print("Arquivos gerados:")
    for caminho in caminhos_saida.values():
        print(f"- {caminho}")
    print()
    print("Analise de triagem concluida.")


if __name__ == "__main__":
    main()
