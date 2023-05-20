import pandas as pd
from utils.data_wrangling import get_seasonal_components
from typing import Optional
import os

class SerieTemporal:
    """Classe para armazenar informações da série temporal.
    frequency: str
        Frequência da série temporal. Pode ser 'H' (horária) ou 'd' (diária).
    """
    def __init__(self,
                 data: pd.DataFrame,
                 y_col: str,
                 date_col_name: str,
                 test_size: int,
                 frequency: str,
                 seasonality: int=24):
        """
        Args:
            data (pd.DataFrame): DataFrame onde o índice é a data-hora.
            y_col (str): Nome da coluna com a variável-alvo.
            date_col_name (str): Nome da coluna (índice) com a data-hora.
            test_size (int): Tamanho da partição de teste.
            frequency (str): Frequência da série. "H" ou "d".
            seasonality (int, optional): Intervalo de tempo cíclico. Padrão de 24 (horas).
        """
        self.data = data
        self.y_col = y_col
        self.full_series = data[y_col]
        self.date_col_name = date_col_name
        self.train = data.iloc[:-test_size][y_col]
        self.horizon = test_size
        self.test = data.iloc[-test_size:][y_col]
        self.frequency = frequency  
        self.seasonality = seasonality
        self.seasonal_components = get_seasonal_components(data.index)

def train_test_split(df: pd.DataFrame, 
                     test: int, 
                     y_col: Optional[str],
                     multivariate=False) -> pd.DataFrame:
    """Função para particionar dataframes uni ou multivariados

    Args:
        df (pd.DataFrame): Dataframe com a série temporal
        test (int): Tamanho da partição de teste, utilizando os dados mais recentes
        multivariate (bool, optional): Dados são multivariados? (Necessário incluir o nome da variável resposta em y_col). Defaults to False.
        y_col (str, optional): Nome da coluna de variável resposta. Defaults to None.

    Returns:
        pd.DataFrame: 4 ou 2 Dataframes de treino e teste, nessa ordem. Se multivariado,
        o resultado é "train_x, train_y, test_x, test_y"; se univariado, "train, test".
    """
    train = df.iloc[:-test,:]
    test = df.iloc[-test:,:]
    if multivariate:
        x_cols = [col for col in df.columns if col != y_col] 
        train_x = train[x_cols]
        train_y = train[y_col]
        test_x = test[x_cols]
        test_y = test[y_col]
        return train_x, train_y, test_x, test_y
    else:
        return train, test
    
def to_supervised_frame(df: pd.DataFrame, 
                        y_col: str, 
                        n_in: int= 1, 
                        n_out: int=1, 
                        dropnan: bool=True) -> pd.DataFrame:
    """Função que transforma dados univariados para o formato tabular com base nos lags.

    Args:
        df (pd.DataFrame): _description_
        n_in (int, optional): _description_. Defaults to 1.
        n_out (int, optional): _description_. Defaults to 1.
        dropnan (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    series = df.loc[:,y_col]
    cols = []
    for i in range(n_in, 0, -1):
        lag = series.shift(i)
        lag.name = f"{y_col}(t-{i})"
        cols.append(lag)
    for i in range(0,n_out+1):
        lag = series.shift(i)
        if i==0:
            lag.name = f"{y_col}(t)"
        else:
            lag.name = f"{y_col}(t+{i})"
        cols.append(lag)
    lags_df = pd.concat(cols, axis=1)
    if dropnan:
        lags_df.dropna(inplace=True)
    return lags_df

def extract_model_cols(data: pd.DataFrame, 
                       model: str, 
                       date_col_name: str="ds") -> pd.DataFrame:
    """Retorna um Dataframe com todas as colunas de um Dataframe que contenha o nome
    de um modelo e também o campo de data. Útil para quando se tem um Dataframe com
    colunas de intervalo de confiança: ['AutoARIMA', 'AutoARIMA-lo-90', 'AutoARIMA-hi-90'].

    Args:
        data (pd.DataFrame): _description_
        model (str): _description_

    Returns:
        pd.DataFrame: Dataframe apenas com as informações do modelo especificado.
    """
    model_cols = data[[x for x in data.columns if model in x or date_col_name in x]]
    return model_cols
