import pandas as pd
import requests

def download_carga_horaria(ano_inicio: int, ano_fim: int, printer: bool=False) -> pd.DataFrame:
    """Função para fazer download dos dados de carga elétrica por subsistema no período de referência em base horária.
       Página: https://dados.ons.org.br/dataset/curva-carga

    Args:
        ano_inicio (int): ano inicial de extração.
        ano_fim (int): ano final de extração.
        printer(bool): rastreia ano que está sendo extraído com print em tela.

    Returns:
        pd.DataFrame: Pandas DataFrame com dados de carga horária entre ano_inicio e ano_fim, tendo como índice a data-hora.
    """

    url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{}.csv"
    # verificar se anos inicial e final estão disponíveis
    get0 = requests.get(url.format(ano_inicio)).status_code # verify = False (autenticação)
    getn = requests.get(url.format(ano_fim)).status_code 
    if (get0 == 200) and (getn == 200): # 200: página (ano) disponível
        # concatenar arquivos de cada ano em um único dataframe
        df = pd.DataFrame()
        for ano in range(ano_inicio, ano_fim + 1):
            if printer:
                print(f"Lendo ano {ano}...")
            df2 = pd.read_csv(url.format(ano), sep = ";")
            df = pd.concat([df, df2])
        df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = '%Y-%m-%d %H:%M:%S')
        df.sort_values(by = "date", inplace = True)
        df.set_index("date", inplace=True)
        return df
    else:
       print("Ano não disponível.")


def download_carga_diaria(ano_inicio: int, ano_fim: int, printer: bool=False) -> pd.DataFrame:
    """Função para fazer download dos dados de carga elétrica por subsistema no período de referência em base diária.
       Página: https://dados.ons.org.br/dataset/carga-energia

    Args:
        ano_inicio (int): ano inicial de extração.
        ano_fim (int): ano final de extração.
        printer(bool): rastreia ano que está sendo extraído com print em tela.

    Returns:
        pd.DataFrame: Pandas DataFrame com dados de carga diária entre ano_inicio e ano_fim, tendo como índice a data.
    """
    url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{}.csv"

    # verificar se anos inicial e final estão disponíveis
    get0 = requests.get(url.format(ano_inicio)).status_code # verify = False (autenticação)
    getn = requests.get(url.format(ano_fim)).status_code 
    if (get0 == 200) and (getn == 200): # 200: página (ano) disponível

        # concatenar arquivos de cada ano em um único dataframe
        df = pd.DataFrame()
        for ano in range(ano_inicio, ano_fim + 1):
            if printer:
                print(f"Lendo ano {ano}...")
            df2 = pd.read_csv(url.format(ano), sep = ";")
            df = pd.concat([df, df2])
        df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = '%Y-%m-%d')
        df.sort_values(by = "date", inplace = True)
        df.set_index("date", inplace=True)
        return df
    
    else:
       print("Ano não disponível.")