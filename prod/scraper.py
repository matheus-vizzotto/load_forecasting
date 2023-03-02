import pandas as pd
import requests
from typing import List
import datetime as dt

class ons_data:
    """Classe destinada à leitura dos dados de carga da ONS
    """
    def __init__(self, freq: str, ano_inicio: int, ano_fim: int, idreg: str=None):
        """
        Args:
            freq (str): frequência da série. ["hourly","daily"]
            ano_inicio (int): ano inicial de extração.
            ano_fim (int): ano final de extração.
            idreg (str): sub-região. ['N', 'NE', 'S', 'SE']
        """
        self.freq = freq
        self.ano_inicio = ano_inicio
        self.ano_fim = ano_fim
        self.idreg = idreg
        self.data = pd.DataFrame()
        self.missing_dates = []
        self.data_dt_inserted = pd.DataFrame()
        self.data_dir = "../../../data/"

    def read(self) -> pd.DataFrame:
        """Função para ler arquivos "csv" já presentes no diretório de dados.

        Args:
            

        Returns:
            pd.DataFrame: série de carga elétrica no período entre ano_inicio e ano_fim.
        """
        if self.freq == "hourly":
            path = "".join([self.data_dir,"hourly_load.csv"])
        elif self.freq == "daily":
            path = "".join([self.data_dir,"daily_load.csv"])
        df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["date"])
        if not self.idreg:
            idreg = df["id_reg"].unique()
        else:
            idreg = [self.idreg]
        df = df[df["id_reg"].isin(idreg)]
        df.set_index("date", inplace=True)
        self.data = df
        return df

    def update(self, printer=False, write: bool=False) -> pd.DataFrame:
        """Função para atualizar os arquivos no diretório de dados.

        Args:
            printer (bool, optional): Informa o progresso do download ano a ano. Defaults to False.
            write (bool, optional): Escreve o arquivo no diretório de dados. Defaults to False.

        Raises:
            Exception: Frequência não está na lista de arquivos disponíveis.

        Returns:
            pd.DataFrame: série atualizada de carga elétrica no período entre ano_inicio e ano_fim.
        """
        if self.freq == "hourly":
            url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{}.csv"
            date_format = "%Y-%m-%d %H:%M:%S"
        elif self.freq == "daily":
            url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{}.csv"
            date_format = "%Y-%m-%d"
        else:
            raise Exception("Frequência não reconhecida. Utilize 'hourly' ou 'daily'.")
        get0 = requests.get(url.format(self.ano_inicio)).status_code # verify = False (autenticação)
        getn = requests.get(url.format(self.ano_fim)).status_code 
        if (get0 == 200) and (getn == 200): # 200: página (ano) disponível
            # concatenar arquivos de cada ano em um único dataframe
            df = pd.DataFrame()
            for ano in range(self.ano_inicio, self.ano_fim + 1):
                if printer:
                    print(f"Lendo ano {ano}...")
                df2 = pd.read_csv(url.format(ano), sep = ";")
                df = pd.concat([df, df2])
            df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = date_format)
            df.sort_values(by = "date", inplace = True)
            df.set_index("date", inplace=True)
            if write:
                full_path = "".join([self.data_dir,f"{self.freq}_load.csv"])
                df.to_csv(full_path, sep=";", decimal=",")
            self.data = df
            return df
        else:
            print("Ano não disponível.")
    
    def check_date_column(self, _freq: str, printer=False) -> List[dt.datetime]:
        """Verifica datas faltantes no intervalo

        Args:
            _freq (str): frequência da série
            printer (bool, optional): informa as datas faltantes em tela. Defaults to False.

        Returns:
            List[dt.datetime]: lista de datas faltantes
        """
        date_col = self.data.reset_index()["date"]
        missing_dates = pd.date_range(date_col.min(), date_col.max(), freq=_freq).difference(date_col)
        missing_list = missing_dates.to_list()
        if printer:
            print("Datas faltantes:\n", missing_list)
        self.missing_dates = missing_list
        return missing_list
    
    def insert_missing_dates(self, printer=False):
        y = self.data.reset_index()
        missing = pd.DataFrame(self.missing_dates, columns=["date"])
        y = pd.concat([y, missing], ignore_index=True)
        y.loc[:,"date"] = pd.to_datetime(y.loc[:,"date"])
        y.set_index("date", inplace=True)
        y.sort_index(inplace=True)
        faltantes = check_date_column(y.index, _freq='h')
        if printer:
            print("Datas faltantes após transformação:", faltantes)
        self.data_dt_inserted = y
        return y


def check_date_column(date_col: List[dt.datetime], _freq: str, printer=False) -> List(dt.datetime):
    """Verifica datas faltantes no intervalo.

    Args:
        date_col (List[dt.datetime]): coluna de datas.
        _freq (str): frequência da coluna de datas.
        printer (bool, optional): traz as datas fatantes em tela. Defaults to False.

    Returns:
        List(dt.datetime): lista de datas faltantes
    """
    missing_dates = pd.date_range(date_col.min(), date_col.max(), freq=_freq).difference(date_col)
    missing_list = missing_dates.to_list()
    if printer:
        print("Datas faltantes:\n", missing_list)
    return missing_list

def insert_missing_dates(x: pd.DataFrame, missing_dates: list, date_column_name: str):
    y = x.reset_index()
    missing = pd.DataFrame(missing_dates, columns=[date_column_name])
    y = pd.concat([y, missing], ignore_index=True)
    y.loc[:,"date"] = pd.to_datetime(y.loc[:,"date"])
    y.set_index("date", inplace=True)
    y.sort_index(inplace=True)
    faltantes = check_date_column(y.index, _freq='h')
    print("Datas faltantes após transformação:", faltantes)
    return y