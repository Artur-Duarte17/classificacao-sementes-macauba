from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================
# SCRIPT 00 - INVENTÃRIO DAS IMAGENS
# ------------------------------------------------------------
# Objetivo:
# - Procurar todas as imagens dentro de dados_originais/imagens
# - Registrar o caminho relativo de cada imagem
# - Verificar se a imagem abre corretamente
# - Salvar relatÃ³rios CSV em saidas/tabelas
#
# Este script NÃƒO treina IA.
# Ele apenas organiza e confere a base de imagens.
# ============================================================


# Pasta raiz do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[2]

# Pasta onde estÃ£o as imagens originais
PASTA_IMAGENS = PASTA_PROJETO / "dados_originais" / "imagens"

# Pasta onde os relatorios serao salvos
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "tabelas" / "01_inventario"

# ExtensÃµes consideradas imagens
EXTENSOES_IMAGEM = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def normalizar_caminho(caminho: Path) -> str:
    """
    Converte um caminho do Windows para um formato padronizado com barra normal.
    Exemplo:
    TESTE 2\\T3\\a1.jpg -> TESTE 2/T3/a1.jpg
    """
    return caminho.as_posix()


def analisar_imagem(caminho_imagem: Path) -> dict:
    """
    Tenta abrir a imagem e extrair informaÃ§Ãµes bÃ¡sicas.

    Retorna:
    - largura
    - altura
    - modo de cor
    - status de leitura
    - mensagem de erro, caso exista
    """
    try:
        with Image.open(caminho_imagem) as img:
            largura, altura = img.size
            modo = img.mode

        return {
            "imagem_valida": True,
            "largura": largura,
            "altura": altura,
            "modo_cor": modo,
            "erro": ""
        }

    except Exception as e:
        return {
            "imagem_valida": False,
            "largura": None,
            "altura": None,
            "modo_cor": "",
            "erro": str(e)
        }


def main():
    """
    FunÃ§Ã£o principal do script.
    """

    print("=" * 60)
    print("INVENTÃRIO DAS IMAGENS")
    print("=" * 60)

    # Garante que a pasta de saÃ­da exista
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    # Confere se a pasta de imagens existe
    if not PASTA_IMAGENS.exists():
        print("ERRO: pasta de imagens nÃ£o encontrada:")
        print(PASTA_IMAGENS)
        return

    print(f"Pasta do projeto: {PASTA_PROJETO}")
    print(f"Pasta de imagens: {PASTA_IMAGENS}")
    print()

    # Busca arquivos com extensÃµes de imagem
    arquivos = [
        caminho
        for caminho in PASTA_IMAGENS.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    ]

    print(f"Total de arquivos de imagem encontrados: {len(arquivos)}")
    print()

    registros = []

    for caminho_absoluto in tqdm(arquivos, desc="Analisando imagens"):
        caminho_relativo = caminho_absoluto.relative_to(PASTA_IMAGENS)

        partes = caminho_relativo.parts

        # Primeira pasta dentro de imagens.
        # Exemplo: Micro-ondas, Piloto, TESTE 2
        experimento = partes[0] if len(partes) >= 1 else ""

        # Pasta imediatamente anterior ao arquivo.
        # Exemplo: T3, T6, Todas juntas
        pasta_pai = caminho_absoluto.parent.name

        info_imagem = analisar_imagem(caminho_absoluto)

        registro = {
            "caminho_relativo": normalizar_caminho(caminho_relativo),
            "caminho_absoluto": str(caminho_absoluto),
            "experimento": experimento,
            "pasta_pai": pasta_pai,
            "nome_arquivo": caminho_absoluto.name,
            "nome_sem_extensao": caminho_absoluto.stem,
            "extensao": caminho_absoluto.suffix.lower(),
            **info_imagem
        }

        registros.append(registro)

    df = pd.DataFrame(registros)

    # ------------------------------------------------------------
    # Salva inventÃ¡rio completo
    # ------------------------------------------------------------
    caminho_inventario = PASTA_SAIDA / "inventario_imagens.csv"
    df.to_csv(caminho_inventario, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Salva imagens problemÃ¡ticas
    # ------------------------------------------------------------
    df_problematicas = df[df["imagem_valida"] == False].copy()

    caminho_problematicas = PASTA_SAIDA / "imagens_problematicas.csv"
    df_problematicas.to_csv(caminho_problematicas, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Salva nomes repetidos
    # Importante porque o mesmo arquivo pode se chamar a1.jpg
    # em pastas diferentes.
    # ------------------------------------------------------------
    contagem_nomes = (
        df.groupby("nome_arquivo")
        .size()
        .reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False)
    )

    df_repetidos = contagem_nomes[contagem_nomes["quantidade"] > 1].copy()

    caminho_repetidos = PASTA_SAIDA / "nomes_repetidos.csv"
    df_repetidos.to_csv(caminho_repetidos, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Salva resumo por experimento e pasta
    # ------------------------------------------------------------
    if len(df) > 0:
        resumo = (
            df.groupby(["experimento", "pasta_pai", "extensao"])
            .size()
            .reset_index(name="quantidade")
            .sort_values(["experimento", "pasta_pai", "extensao"])
        )
    else:
        resumo = pd.DataFrame(columns=["experimento", "pasta_pai", "extensao", "quantidade"])

    caminho_resumo = PASTA_SAIDA / "resumo_inventario.csv"
    resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Exibe resumo no terminal
    # ------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Total de imagens encontradas: {len(df)}")
    print(f"Imagens vÃ¡lidas: {df['imagem_valida'].sum() if len(df) > 0 else 0}")
    print(f"Imagens problemÃ¡ticas: {len(df_problematicas)}")
    print(f"Nomes de arquivos repetidos: {len(df_repetidos)}")
    print()
    print("Arquivos gerados:")
    print(f"- {caminho_inventario}")
    print(f"- {caminho_problematicas}")
    print(f"- {caminho_repetidos}")
    print(f"- {caminho_resumo}")

    print()
    print("InventÃ¡rio concluÃ­do.")


if __name__ == "__main__":
    main()


