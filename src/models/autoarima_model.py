import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utils.data_wrangling import prepare_statsforecast_df

def auto_arima_model(df: pd.DataFrame, h_: int, season_length=24, frequency: str='H', level=[90]) -> pd.DataFrame: 
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        h_ (int): _description_
        season_length (_type_): _description_
        frequency (str): _description_
        level (list, optional): _description_. Defaults to [90].

    Returns:
        pd.DataFrame: _description_
    """
    df_sf = prepare_statsforecast_df(df, "hourly_load")
    sf = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=frequency
    )
    sf.fit(df_sf)
    forecasts_df = sf.forecast(h=h_, level=level)
    forecasts_df_final = forecasts_df[["ds", "AutoARIMA"]]
    return forecasts_df_final
