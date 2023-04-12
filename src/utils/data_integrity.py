import pandas as pd
import numpy as np
import os

def check_date_range(data: pd.DataFrame, date_col_name: str, frequency: str, date_is_index: bool):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        date_col_name (str): _description_
        frequency (str): ["min", "h", "d", "M"]
        date_is_index (bool): _description_
    """
    if date_is_index:
        date_col = data.reset_index()[date_col_name]
    else:
        date_col = data[date_col_name]
    _dt_range = pd.date_range(date_col.min(), date_col.max(), freq=frequency)
    missing_dates_ = _dt_range.difference(date_col).to_list()
    if missing_dates_:
        return "Datas faltantes: ", missing_dates_
    else:
        return "Sem datas faltantes.", None
    
def input_missing_dates(data: pd.DataFrame, date_col_name: str, missing_dates_: list, date_is_index: bool) -> pd.DataFrameS:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        date_col_name (str): _description_
        date_is_index (bool): _description_

    Returns:
        pd.DataFrameS: _description_
    """
    if date_is_index:
        y = data.reset_index()
    else:
        y = data
    missing = pd.DataFrame(missing_dates_, columns=[f"{date_col_name}"])
    y = pd.concat([y, missing], ignore_index=True).sort_values(by=date_col_name)
    return y