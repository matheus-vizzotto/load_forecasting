import pandas as pd
import numpy as np
import os
import time
import datetime as dt

def check_date_range(data: pd.DataFrame, 
                     date_col_name: str, 
                     frequency: str):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        date_col_name (str): _description_
        frequency (str): ["min", "h", "d", "M"]
        date_is_index (bool): _description_
    """
    # if date_is_index:
    #     date_col = data.reset_index()[date_col_name]
    # else:
    date_col = data[date_col_name]
    _dt_range = pd.date_range(date_col.min(), date_col.max(), freq=frequency)
    missing_dates_ = _dt_range.difference(date_col).to_list()
    if missing_dates_:
        return f"Datas faltantes: {len(missing_dates_)}", missing_dates_
    else:
        return "Sem datas faltantes.", None
    
def input_missing_dates(data_: pd.DataFrame, 
                        date_col_name: str, 
                        freq: str, 
                        date_is_index: bool) -> pd.DataFrame:
    """Verifica se há datas faltantes no dataframe e, no caso positivo, as inclui no original, deixando as
    demais colunas com NaN.

    Args:
        data (pd.DataFrame): DataFrame com coluna de data
        date_col_name (str): Nome da coluna de data
        date_is_index (bool): A coluna de data está como índice?

    Returns:
        pd.DataFrameS: _description_
    """
    if date_is_index:
        y = data_.reset_index()
    else:
        y = data_
    msg, missing_dates_ = check_date_range(data=y, date_col_name=date_col_name, frequency=freq)
    missing = pd.DataFrame(missing_dates_, columns=[f"{date_col_name}"])
    y = pd.concat([y, missing], ignore_index=True).sort_values(by=date_col_name)
    return msg, y

def check_outliers(data: pd.Series, 
                   method='desvpad') -> pd.Series:
    """_summary_

    Args:
        data (pd.Series): _description_
        method (str, optional): _description_. Defaults to 'desvpad'.

    Returns:
        pd.Series: _description_
    """
    if method == 'desvpad':
        window = 24*7
        n_std = 2
        outlier = np.where( (data > data+(n_std*data.rolling(window).std())) | (data < data-(n_std*data.rolling(window).std()),1,0) )
    return outlier

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        delta_time = dt.timedelta(seconds=total_time)
        ref_date = dt.datetime(2023, 4, 29, 0, 0, 0)  # fixed reference date
        time_str = (ref_date + delta_time).strftime("%H:%M:%S")
        print(f"\tTempo de execução para {func.__name__}: {time_str}")
        return result
    return wrapper