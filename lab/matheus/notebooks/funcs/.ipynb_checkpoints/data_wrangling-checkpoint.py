import pandas as pd
import numpy as np
import requests
from typing import List
import datetime as dt
import matplotlib.pyplot as plt

class ons_data:
    """Classe destinada à leitura dos dados de carga da ONS
    """
    def __init__(self, freq: str, ano_inicio: int, ano_fim: int, idreg: str=None):
        """
        Args:
            freq (str): frequência da série. ["h","d"]
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
        self.datas_estranhas = []
        self.seasonal_components = pd.DataFrame()
        #self.data_dt_inserted = pd.DataFrame()
        #self.data_treated = pd.DataFrame()
        self.data_dir = "../../../data/"

    def read(self) -> pd.DataFrame:
        """Função para ler arquivos "csv" já presentes no diretório de dados.

        Args:
            

        Returns:
            pd.DataFrame: série de carga elétrica no período entre ano_inicio e ano_fim.
        """
        if self.freq == "h":
            path = "".join([self.data_dir,"hourly_load.csv"])
        elif self.freq == "d":
            path = "".join([self.data_dir,"daily_load.csv"])
        df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["date"])
        if not self.idreg:
            idreg = df["id_reg"].unique()
        else:
            idreg = [self.idreg]
        df = df[df["id_reg"].isin(idreg)]
        df.set_index("date", inplace=True)
        self.data = df
        #return df

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
        if self.freq == "h":
            url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{}.csv"
            date_format = "%Y-%m-%d %H:%M:%S"
        elif self.freq == "d":
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
            if not self.idreg:
                idreg = df["id_reg"].unique()
            else:
                idreg = [self.idreg]
            df = df[df["id_reg"].isin(idreg)]
            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = date_format)
            df.sort_values(by = "date", inplace = True)
            df.set_index("date", inplace=True)
            if write:
                full_path = "".join([self.data_dir,f"{self.freq}_load.csv"])
                df.to_csv(full_path, sep=";", decimal=",")
            self.data = df
            #return df
        else:
            print("Ano não disponível.")
    
    def check_date_column(self, printer=True) -> List[dt.datetime]:
        """Verifica datas faltantes no intervalo

        Args:
            _freq (str): frequência da série
            printer (bool, optional): informa as datas faltantes em tela. Defaults to False.

        Returns:
            List[dt.datetime]: lista de datas faltantes
        """
        date_col = self.data.reset_index()["date"]
        _dt_range = pd.date_range(date_col.min(), date_col.max(), freq=self.freq)
        missing_dates_ = _dt_range.difference(date_col)
        missing_list = missing_dates_.to_list()
        dts_extras = date_col[~(date_col.isin(_dt_range))].to_list()
        if printer:
            print("Datas faltantes (incluir):\n", missing_list)
            print("Datas estranhas (retirar):\n", dts_extras)
        self.datas_estranhas = dts_extras
        self.missing_dates = missing_list
        #return missing_list
    
    def correct_dates(self, printer=False):
        y = self.data
        if printer:
            print(f"Inserindo {len(self.missing_dates)} datas.\nRetirando {len(self.datas_estranhas)} datas.")
        if self.datas_estranhas:
            y.drop(self.datas_estranhas, axis=0, inplace=True)
        y.reset_index(inplace=True)
        missing = pd.DataFrame(self.missing_dates, columns=["date"])
        y = pd.concat([y, missing], ignore_index=True)
        y.loc[:,"date"] = pd.to_datetime(y.loc[:,"date"])
        y.set_index("date", inplace=True)
        y.sort_index(inplace=True)
        self.data = y
        #return y
    
    def remove_outliers(self: pd.DataFrame) -> pd.DataFrame:
        y = self.data
        y.loc[:,"load_mwmed"] = np.where(y.loc[:,"load_mwmed"] <= 0, np.nan, y.loc[:,"load_mwmed"])
        self.data = y

    def fill_na(self, _method: str, printer=False):
        """Preenche valores vazios.
        Args:
            _method (str): método para preencher os valores vazios. ["linear", "nearest", "spline", "polynomial"]
        """
        data = self.data
        data.loc[:, "id_reg"] = data.loc[:, "id_reg"].ffill().bfill() 
        data.loc[:, "desc_reg"] = data.loc[:, "desc_reg"].ffill().bfill() 
        data.loc[:, "load_mwmed"] = data.loc[:, "load_mwmed"].interpolate(method=_method)
        if printer:
            print("Valores vazios restantes:")
            print(data.isna().sum())
        self.data = data
        return data

    def get_data_description(self, plot=False):
        data = self.data
        print(data.describe(include='all'))
        print(data.info())
        print("Valores vazios:")
        n_missing = data["load_mwmed"].isna().sum()
        print(data.isna().sum())
        print("\nValor mínimo:", data["load_mwmed"].min())
        print("Valor máximo:", data["load_mwmed"].max())
        if plot==True:
            missing_idx = data[data["load_mwmed"].isna()].index
            fig, ax = plt.subplots(figsize=(15,5))
            plt.plot(pd.to_datetime(data.index), data["load_mwmed"])
            for data_faltante in missing_idx:
                plt.axvline(x=data_faltante, color='red', linestyle='--')
            plt.title(f"Valores vazios: {n_missing}")
            plt.show()

    def get_seasonal_components(self):
        x0 = self.data
        x = x0.reset_index()
        y = pd.DataFrame()
        y["data"] = x["date"]
        y["ano"] = x["date"].dt.year
        y["trimestre"] = x["date"].dt.quarter
        y["mes"] = x["date"].dt.month
        y["semana_ano"] = x["date"].dt.isocalendar().week
        y["dia"] = x["date"].dt.day
        y["dia_ano"] = x["date"].dt.dayofyear
        y["dia_semana"] = x["date"].dt.weekday + 1    # 1: segunda-feira; 7: domingo
        y["hora"] = x["date"].dt.hour
        y["apagao"] = x["date"].dt.year.apply(lambda x: 1 if x in [2001, 2002] else 0) # apagão de 2001 e 2002
        self.seasonal_components = y

    
def pipeline(x, update=False) -> pd.DataFrame:
    """Função que aplica o tratamento de dados na classe ons_data.

    Args:
    Returns:
        pd.DataFrame: dados ajustados
    """
    data = x
    if update:
        data.update()
    else:
        data.read()
    #df0 = data.data
    data.check_date_column(printer=False)
    data.correct_dates(printer=False)
    #data.get_data_description(plot=False)
    data.remove_outliers()
    data.fill_na(_method="linear")
    data.get_seasonal_components()
    df = data.data
    #print(df.describe())
    return df
    #df.describe(include=category)