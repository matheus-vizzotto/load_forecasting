import datetime as dt
import os
import shutil
from io import BytesIO
from typing import List
from typing import Optional
from zipfile import ZipFile
import utils.logger as lg
from utils.logger import timer_decorator, general_data_info_log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from paths import PATHS
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#import mypy

class ons_data:
    """Classe destinada à leitura dos dados de carga da ONS
    """
    def __init__(self, 
                 freq: str, 
                 ano_inicio: int, 
                 ano_fim: int, 
                 #idreg: str=None):
                 idreg: Optional[str]):
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
        self.missing_dates : List[dt.datetime] = []
        self.datas_estranhas : List[dt.datetime] = []
        self.seasonal_components = pd.DataFrame()
        #self.data_dt_inserted = pd.DataFrame()
        #self.data_treated = pd.DataFrame()
        #self.data_dir = "../data/03_processed/"
        self.data_dir = PATHS['processed_data']

    def read(self) -> pd.DataFrame:
        """Função para ler arquivos "csv" já presentes no diretório de dados.

        Args:
            

        Returns:
            pd.DataFrame: série de carga elétrica no período entre ano_inicio e ano_fim.
        """
        if self.freq == "h":
            path = "".join([self.data_dir,"/h_load.parquet"])
        elif self.freq == "d":
            path = "".join([self.data_dir,"/d_load.parquet"])
        #df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["date"])
        df = pd.read_parquet(path)
        if not self.idreg:
            idreg = df["id_reg"].unique()
        else:
            idreg = [self.idreg]
        df = df[df["id_reg"].isin(idreg)]
        #df.set_index("date", inplace=True)
        self.data = df
        #return df
    @timer_decorator
    def update(self, 
               printer=False, 
               write: bool=False) -> pd.DataFrame:
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
                    print(f"\tLendo ano {ano}...")
                df2 = pd.read_csv(url.format(ano), sep = ";")
                df = pd.concat([df, df2])
            df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
            df = df.astype({
                    "date": "datetime64[ns]",
                    "id_reg": "category",
                    "desc_reg": "category",
                    "load_mwmed": "float64"
                    })
            if not self.idreg:
                idreg = df["id_reg"].unique()
            else:
                idreg = [self.idreg]
            df = df[df["id_reg"].isin(idreg)]
            #df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = date_format)
            df.sort_values(by = "date", inplace = True)
            df.set_index("date", inplace=True)
            if write:
                #full_path = "".join([self.data_dir,f"{self.freq}_load.csv"])
                full_path = "".join([self.data_dir,f"{self.freq}_load.parquet"])
                full_path = os.path.join(self.data_dir, f"{self.freq}_load.parquet")
                df.to_parquet(full_path)
                #df.to_csv(full_path, sep=";", decimal=",")
            self.data = df
            #return df
        else:
            print("Ano não disponível.")
    
    def check_date_column(self, 
                          printer=True) -> List[dt.datetime]:
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
        return missing_list
    
    def correct_dates(self, 
                      printer=False):
        """_summary_

        Args:
            printer (bool, optional): _description_. Defaults to False.
        """
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
        """_summary_

        Args:
            self (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        y = self.data
        y.loc[:,"load_mwmed"] = np.where(y.loc[:,"load_mwmed"] <= 0, np.nan, y.loc[:,"load_mwmed"])
        self.data = y

    def fill_na(self, 
                _method: str, 
                printer=False):
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

    def get_data_description(self, 
                             plot=False):
        """_summary_

        Args:
            plot (bool, optional): _description_. Defaults to False.
        """
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
        """_summary_
        """
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

    
def pipeline(x, 
             update=False) -> pd.DataFrame:
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

def prepare_statsforecast_df(x: pd.Series, 
                             unique_id: str, 
                             date_is_index=True,) -> pd.DataFrame:
    """Função para transformar um DataFrame para o formato utilizado pelos algoritmos do pacote statsforecast (Nixtla)

    Args:
        x (pd.DataFrame): dataframe original.
        unique_id (str): identificador da série (arbitrário).

    Returns:
        pd.DataFrame: dataframe no formato necessário.
    """
    if date_is_index:
        df2 = x.reset_index()
    else:
        df2 = x[["date", "load_mwmed"]]
    df2.columns = ["ds", "y"]
    df2["unique_id"] = unique_id
    return df2


class inmet_data:
    def __init__(self, 
                 ano_inicio: int, 
                 ano_fim: int):
        """_summary_

        Args:
            ano_inicio (int): _description_
            ano_fim (int): _description_
        """
        self.ano_inicio = ano_inicio
        self.ano_fim = ano_fim
        self.data = pd.DataFrame()
        self.freq = 'h'
        self.temp_data_dir = PATHS['interim_data']
        self.processed_data_dir = PATHS['processed_data']
        #self.url = f'https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip'
        self.columns_dict = {
                            'DATA (YYYY-MM-DD)': 'data',
                            'Data': 'data',
                            'HORA (UTC)': 'hora',
                            'Hora UTC': 'hora',
                            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_horario_mm',
                            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_horario_mm',
                            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b',
                            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'pressao_atmosferica_max_na_hora_ant_aut_m_b',
                            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'pressao_atmosferica_min_na_hora_ant_aut_m_b',
                            'RADIACAO GLOBAL (KJ/m²)': 'radiacao_global_kj_m',
                            'RADIACAO GLOBAL (Kj/m²)': 'radiacao_global_kj_m',
                            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temperatura_do_ar_bulbo_seco_horaria_c',
                            'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'temperatura_do_ponto_de_orvalho_c',
                            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'temperatura_maxima_na_hora_ant_aut_c',
                            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'temperatura_minima_na_hora_ant_aut_c',
                            'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'temperatura_orvalho_max_na_hora_ant_aut_c',
                            'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'temperatura_orvalho_min_na_hora_ant_aut_c',
                            'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'umidade_rel_max_na_hora_ant_aut_%',
                            'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'umidade_rel_min_na_hora_ant_aut_%',
                            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade_relativa_do_ar_horaria_%',
                            'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'vento_direcao_horaria_gr_gr',
                            'VENTO, RAJADA MAXIMA (m/s)': 'vento_rajada_maxima_m_s',
                            'VENTO, VELOCIDADE HORARIA (m/s)': 'vento_velocidade_horaria_m_s',
                            'ESTACAO': 'estacao',
                            'UF': 'uf',
                            'REGIAO': 'regiao',
                        }
        self.col_types = {
                        'precipitacao_total_horario_mm': float,
                        'pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b': float,
                        'pressao_atmosferica_max_na_hora_ant_aut_m_b': float,
                        'pressao_atmosferica_min_na_hora_ant_aut_m_b': float,
                        'radiacao_global_kj_m': float,
                        'temperatura_do_ar_bulbo_seco_horaria_c': float,
                        'temperatura_do_ponto_de_orvalho_c': float,
                        'temperatura_maxima_na_hora_ant_aut_c': float,
                        'temperatura_minima_na_hora_ant_aut_c': float,
                        'temperatura_orvalho_max_na_hora_ant_aut_c': float,
                        'temperatura_orvalho_min_na_hora_ant_aut_c':float,
                        'umidade_rel_max_na_hora_ant_aut_%': float,
                        'umidade_rel_min_na_hora_ant_aut_%': float,
                        'umidade_relativa_do_ar_horaria_%': float,
                        'vento_direcao_horaria_gr_gr': float,
                        'vento_rajada_maxima_m_s': float,
                        'vento_velocidade_horaria_m_s': float,
                        'estacao': 'category',
                        'uf': 'category',
                        'regiao': 'category'
                    } 
        
    def ajusta_hora(self, 
                    x) -> str:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            str: _description_
        """
        if ":" in x:
            y = x
        elif "UTC" in x:
            y = x[:2] + ":" + x[2:4]
        else:
            y = None
        return y

    def converte_data(self, 
                      x) -> dt.datetime:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            dt.datetime: _description_
        """
        try:
            y = pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S')
        except:
            y = pd.to_datetime(x, format = '%Y/%m/%d %H:%M:%S')
        return y
    
    def write_parquet(self, 
                      df_, 
                      espec=None):
        """_summary_

        Args:
            df_ (_type_): _description_
            espec (_type_, optional): _description_. Defaults to None.
        """
        if espec:
            ending = "".join(["_", str(espec)])
        else:
            ending = ""
        #df_.to_parquet(f"../data/processed/inmet/inmet_data{ending}.parquet")
        file_path = os.path.join(self.temp_data_dir, "inmet/inmet_data{}.parquet".format(ending))
        df_.to_parquet(file_path)

    def correct_dates(self, 
                      estacao: str, 
                      data: pd.DataFrame, 
                      missing_dates: list, 
                      datas_estranhas=None, 
                      printer: bool=False):
        """_summary_

        Args:
            estacao (str): _description_
            data (pd.DataFrame): _description_
            missing_dates (list): _description_
            datas_estranhas (_type_, optional): _description_. Defaults to None.
            printer (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        y = data
        if printer:
            print(f"Inserindo {len(missing_dates)} datas.\nRetirando {len(datas_estranhas)} datas.")
        if datas_estranhas:
            y.drop(datas_estranhas, axis=0, inplace=True)
        y.reset_index(inplace=True)
        missing = pd.DataFrame(missing_dates, columns=["data_hora"])
        y = pd.concat([y, missing], ignore_index=True)
        y.loc[:,"estacao"] = estacao
        y.loc[:,"data_hora"] = pd.to_datetime(y.loc[:,"data_hora"])
        y.sort_values(by="data_hora",inplace=True)
        #y.set_index(["estacao","data_hora"], inplace=True)
        return y

    def check_date_column(self, 
                          estacao, 
                          data_, 
                          printer=True, 
                          ano=None) -> List[dt.datetime]:
        """Verifica datas faltantes no intervalo

        Args:
            _freq (str): frequência da série
            printer (bool, optional): informa as datas faltantes em tela. Defaults to False.

        Returns:
            List[dt.datetime]: lista de datas faltantes
        """
        date_col = data_["data_hora"]
        _dt_range = pd.date_range(date_col.min(), date_col.max(), freq=self.freq)
        missing_dates_ = _dt_range.difference(date_col)
        missing_list = missing_dates_.to_list()
        dts_extras = date_col[~(date_col.isin(_dt_range))].to_list()
        if printer and (missing_list or dts_extras):
            print("Datas faltantes (incluir):\n", missing_list)
            print("Datas estranhas (retirar):\n", dts_extras)
        datas_estranhas = dts_extras
        missing_dates = missing_list
        lg.log_dates(estacao=estacao, missing_dates=missing_dates, datas_estranhas=datas_estranhas, ano=ano)
        return self.correct_dates(data=data_, estacao = estacao, missing_dates=missing_dates, datas_estranhas=datas_estranhas)
    
    def fill_na(self, 
                col: pd.Series) -> pd.Series:
        """_summary_

        Args:
            col (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        if ((col.dtype.kind in 'iu') or (col.dtype.kind in 'f')): # coluna é de int ou float
            roll_mean = col.rolling(window=30, min_periods=1).mean()
            col = col.fillna(roll_mean).fillna(col.mean())#.fillna(method="bfill")
        else:
            col = col.fillna(method="ffill").fillna(method="bfill")
        return col
    
    @timer_decorator
    def download(self) -> None:
        """Função que cria um diretório "inmet" no diretório atual e salva os arquivos tratados de cada ano nela
        para depois serem unificados e salvos pelo método "build_database".
        """
        #temp_path = "../data/interim/"
        #if "inmet" in os.listdir(temp_path):
        if "inmet" in os.listdir(self.temp_data_dir):
            print("\t*Diretório 'inmet' já presente na pasta de dados.")
            pass
        else:
            print("\t*Criando diretório 'inmet' na pasta de dados...")
            #os.mkdir(temp_path + "/inmet")
            os.mkdir(self.temp_data_dir + "/inmet")
        df = pd.DataFrame()
        for ano in range(self.ano_inicio, self.ano_fim+1):
            print(f"\tTrabalhando nos dados de {ano}...")
            path = f'https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip'
            r = requests.get(path, verify = False)
            files = ZipFile(BytesIO(r.content))
            arquivos = [file for file in files.namelist() if file.lower().endswith(".csv")]
            df01 = pd.DataFrame()
            for arquivo in arquivos:
                #print("\n\n")
                info = pd.read_csv(files.open(arquivo), sep = ";", encoding = "latin-1", nrows=7, header = None)
                #info2 = {line[1][0]: line[1][1] for line in info.iterrows()}
                df02 = pd.read_csv(files.open(arquivo),  sep = ";", encoding = "latin-1", skiprows = 8)#, nrows=4)
                #df02.drop(2,axis=0,inplace=True) #teste para ver se check_date_column funcion#
                df02.rename(columns=self.columns_dict, inplace=True)
                estacao = info.iloc[2,1]
                regiao = info.iloc[0,1]
                if regiao != "S":  # FILTRAR APENAS REGIÃO SUL
                    continue
                df02["estacao"] = estacao
                df02["uf"] = info.iloc[1,1]
                df02["regiao"] = regiao
                for col in df02.columns:
                    df02.loc[:,col] = df02.loc[:,col].replace(",",".", regex=True)
                df02 = df02.astype(self.col_types)
                df02.loc[:, "hora"] = df02.loc[:, "hora"].apply(self.ajusta_hora)
                df02["data_hora"] = df02["data"] + " " + df02["hora"]
                df02.loc[:, "data_hora"] = df02.loc[:, "data_hora"].apply(self.converte_data) 
                if "Unnamed: 19" in df02.columns:
                    df02.drop(["Unnamed: 19"], axis=1, inplace=True)
                df02.drop(["data", "hora"], axis=1, inplace=True)
                df02.replace(-9999, np.nan, inplace=True)
                lg.log_data_info(estacao=estacao, ano=ano, nans=df02.isna().sum().sum())
                df02 = self.check_date_column(estacao=estacao, data_=df02, ano=ano)
                for col in df02.columns:
                    df02.loc[:,col] = self.fill_na(df02[col])
                #df02 = self.check_date_column(estacao=estacao, data_=df02, ano=ano)
                lg.log_data_info(estacao=estacao, ano=ano, nans=df02.isna().sum().sum())
                #print("DATA PARA CHECK:", df02.iloc[2])
                df01 = pd.concat([df01, df02])
            df01.sort_values(by=["estacao", "data_hora"])
            self.write_parquet(df01, espec=ano)

    @timer_decorator
    def build_database(self, 
                       delete_partial_data=False):
        """_summary_

        Args:
            delete_partial_data (bool, optional): _description_. Defaults to True.
        """
        #temp_path = "../data/interim/inmet/"
        temp_path = os.path.join(self.temp_data_dir, "inmet/")
        files = ["".join([temp_path, file]) for file in os.listdir(temp_path)]
        print("AGREGAÇÃO DOS DADOS")
        df = pd.DataFrame()
        for file in files:
            file_name_sub = file.split("/")[-1]
            print(f"\tConcatenando arquivo: {file_name_sub}")
            df0 = pd.read_parquet(file)
            df0 = df0.astype(self.col_types)
            df = pd.concat([df,df0])
        #df.set_index(["estacao","data_hora"], inplace=True)
        #df.to_parquet("../data/inmet_data.parquet")
        file_path = os.path.join(self.processed_data_dir, "inmet_data.parquet")
        df.to_parquet(file_path)
        print("\t*Arquivo 'inmet_data.parquet' salvo no diretório atual. Deletando pasta 'inmet'.")
        if delete_partial_data:
            shutil.rmtree(temp_path)
        else:
            pass

    def read_parquet(self, final=True):
        """_summary_

        Args:
            start (int, optional): _description_. Defaults to self.data_inicio.
            end (int, optional): _description_. Defaults to self.data_fim.

        Returns:
            _type_: _description_
        """
        path = None
        if final:
            file_path = os.path.join(self.processed_data_dir, "inmet_data_agg.parquet")
            self.data = pd.read_parquet(file_path)
        else:
            file_path = os.path.join(self.processed_data_dir, "inmet_data.parquet")
            df = pd.read_parquet(file_path)
            #df = pd.read_parquet("../data/processed/inmet_data.parquet")
            df = df.astype(self.col_types)
            df.set_index(["estacao","data_hora"], inplace=True)
            df.sort_index(inplace=True)
            df.drop("index", axis=1, inplace=True)
            idx = pd.IndexSlice
            inicio = f"{self.ano_inicio}"
            fim = f"{self.ano_fim}"
            df2 = df.loc[idx[:, inicio:fim], :]
            df2 = df[df2.regiao == "S"]
            self.data = df
        #return df2

    def aggregate_by_hour(self, 
                          write=True):
        """_summary_

        Args:
            write (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        agg_dict = {
                'precipitacao_total_horario_mm': 'sum',
                'pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b': 'mean',
                'pressao_atmosferica_max_na_hora_ant_aut_m_b': 'mean',
                'pressao_atmosferica_min_na_hora_ant_aut_m_b': 'mean', 
                'radiacao_global_kj_m': 'mean',
                'temperatura_do_ar_bulbo_seco_horaria_c': 'mean',
                'temperatura_do_ponto_de_orvalho_c': 'mean',
                'temperatura_maxima_na_hora_ant_aut_c': 'mean',
                'temperatura_minima_na_hora_ant_aut_c': 'mean',
                'temperatura_orvalho_max_na_hora_ant_aut_c': 'mean',
                'temperatura_orvalho_min_na_hora_ant_aut_c': 'mean',
                'umidade_rel_max_na_hora_ant_aut_%': 'mean',
                'umidade_rel_min_na_hora_ant_aut_%': 'mean', 
                'umidade_relativa_do_ar_horaria_%': 'mean',
                'vento_direcao_horaria_gr_gr': 'mean', 
                'vento_rajada_maxima_m_s': 'mean',
                'vento_velocidade_horaria_m_s': 'mean' 
            }
        nans = self.data.reset_index().isna().sum()
        lg.general_data_info_log(f"Valores vazios antes de agregar: \n{nans}")
        df = self.data.reset_index().groupby("data_hora").agg(agg_dict)
        if write:
            file_path = os.path.join(self.processed_data_dir, "inmet_data_agg.parquet")
            df.to_parquet(file_path)
            #df.to_parquet("../data/processed/inmet_data_agg.parquet")
        else:
            return df
    
def get_seasonal_components(date_col: pd.Series, frequency="H") -> pd.DataFrame:
    """Extrai os componentes sazonais da série diária ou horária

    Args:
        date_col (pd.Series): _description_
        frequency (str, optional): Frequência da série. Defaults to "H". ["H", "d"]

    Returns:
        pd.DataFrame: _description_
    """
    if isinstance(date_col, pd.DatetimeIndex):
        date_col = date_col.to_series()
    y = pd.DataFrame()
    y["data"] = date_col
    y["ano"] = date_col.dt.year
    y["trimestre"] = date_col.dt.quarter
    y["mes"] = date_col.dt.month
    if (frequency=="d") or (frequency=="H"):
        y["semana_ano"] = date_col.dt.isocalendar().week
        y["dia"] = date_col.dt.day
        y["dia_ano"] = date_col.dt.dayofyear
        y["dia_semana"] = date_col.dt.weekday + 1    # 1: segunda-feira; 7: domingo
    if frequency=="H":
        y["hora"] = date_col.dt.hour
    #y["apagao"] = date_col.dt.year.apply(lambda x: 1 if x in [2001, 2002] else 0) # apagão de 2001 e 2002
    return y



# Função para pipeline de dados
# class DataPipeline: 
#     def __init__(self, data_source, transformation_funcs=None): 
#         self.data_source = data_source 
#         self.transformation_funcs = transformation_funcs or [] 
    
#     def add_transformation(self, func): 
#         self.transformation_funcs.append(func) 
        
#     def run_pipeline(self): 
#         data = pd.read_csv(self.data_source) 
#         for func in self.transformation_funcs: 
#             data = func(data) 
#         return data
# def filter_data(data): 
#     return data[data['col1'] > 0] 
# def aggregate_data(data): 
#     return data.groupby('col2').sum() 
# pipeline = DataPipeline('data.csv') 
# pipeline.add_transformation(filter_data) 
# pipeline.add_transformation(aggregate_data) 
# transformed_data = pipeline.run_pipeline() 
# print(transformed_data.head())