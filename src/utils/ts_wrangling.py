import pandas as pd
import os

def train_test_split(df: pd.DataFrame, test: int, multivariate=False, y_col: str=None) -> pd.DataFrame:
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
    
def to_supervised_frame(df: pd.DataFrame, y_col: str, n_in: int= 1, n_out: int=1, dropnan: bool=True) -> pd.DataFrame:
    """Função que transforma dados univariados para o formato tabular com base nos lags.

    Args:
        df (pd.DataFrame): _description_
        n_in (int, optional): _description_. Defaults to 1.
        n_out (int, optional): _description_. Defaults to 1.
        dropnan (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    pass