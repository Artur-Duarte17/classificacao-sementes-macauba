from pathlib import Path
import json

import pandas as pd


# ============================================================
# SCRIPT 32 - GERAR SCORES CANDIDATOS DA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Transformar a tabela integrada em formato longo
# - Registrar scores individuais como analises secundarias
# - Registrar o consenso pre-especificado sem escolher por teste externo
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TRIAGEM = PASTA_PROJETO / "saidas" / "tabelas" / "08_triagem"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM / "tabela_integrada_triagem.csv"
CAMINHO_SCORES = PASTA_TRIAGEM / "scores_candidatos_triagem.csv"
CAMINHO_MANIFESTO = PASTA_TRIAGEM / "manifesto_scores_triagem.json"


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


def colunas_probabilidade(tabela: pd.DataFrame) -> list[str]:
    return sorted(coluna for coluna in tabela.columns if coluna.startswith("prob_"))


def modelo_a_partir_coluna(coluna: str) -> str:
    return coluna.removeprefix("prob_")


def criar_scores(tabela: pd.DataFrame) -> pd.DataFrame:
    colunas_prob = colunas_probabilidade(tabela)
    if not colunas_prob:
        raise ValueError("Tabela integrada nao contem colunas prob_*.")

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
    registros = []
    for coluna in colunas_prob:
        modelo_slug = modelo_a_partir_coluna(coluna)
        parcial = tabela[colunas_base + [coluna]].copy()
        parcial = parcial.rename(columns={coluna: "score_contaminacao"})
        parcial["estrategia_score"] = f"individual_{modelo_slug}"
        parcial["tipo_estrategia"] = "individual_descritiva"
        parcial["modelo_referencia"] = modelo_slug
        parcial["participa_consenso_oficial"] = True
        parcial["estrategia_oficial"] = "consenso_pre_especificado"
        parcial["criterio_definido_antes_avaliacao"] = True
        parcial["usa_resultado_externo_para_selecao"] = False
        registros.append(parcial)

    scores = pd.concat(registros, ignore_index=True)
    scores["score_contaminacao"] = pd.to_numeric(
        scores["score_contaminacao"],
        errors="raise",
    )
    return scores


def main() -> None:
    print("=" * 70)
    print("GERANDO SCORES CANDIDATOS DA TRIAGEM")
    print("=" * 70)

    tabela = ler_csv_obrigatorio(CAMINHO_TABELA_INTEGRADA)
    scores = criar_scores(tabela)
    gravar_csv_atomico(scores, CAMINHO_SCORES)
    gravar_json_atomico(
        {
            "protocolo": "triagem_preventiva_crossfit",
            "origem": caminho_relativo(CAMINHO_TABELA_INTEGRADA),
            "arquivo_saida": caminho_relativo(CAMINHO_SCORES),
            "estrategia_oficial": "consenso_pre_especificado",
            "criterio_definido_antes_avaliacao": True,
            "usa_resultado_externo_para_selecao": False,
            "estrategias_individuais": "analises_secundarias_descritivas",
            "media_maximo_probabilidades": "excluidos_por_falta_de_probabilidades_internas_por_amostra",
            "n_linhas": int(len(scores)),
            "n_amostras": int(tabela["nome_arquivo"].nunique()),
            "n_modelos": int(len(colunas_probabilidade(tabela))),
        },
        CAMINHO_MANIFESTO,
    )

    print(f"Scores gerados: {len(scores)}")
    print(f"Amostras: {tabela['nome_arquivo'].nunique()}")
    print(f"Modelos: {len(colunas_probabilidade(tabela))}")
    print(f"- {CAMINHO_SCORES}")
    print(f"- {CAMINHO_MANIFESTO}")


if __name__ == "__main__":
    main()
