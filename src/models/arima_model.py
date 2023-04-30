import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from utils.data_wrangling import prepare_statsforecast_df
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from datetime import datetime
import os
import joblib
from paths import PATHS

FORECASTS_FIG_DIR = PATHS['forecasts_figs']


def auto_arima_model(data: pd.Series,
                     h_: int, 
                     season_length=24, 
                     frequency: str='H', 
                     level=[90],
                     ts_name_id= 'hourly_load',
                     test: Optional[pd.Series]=None,
                     write: bool=True,
                     fcs_dir: Optional[str]=None,
                     save_model: bool=True) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.Series): _description_
        h_ (int): _description_
        season_length (int, optional): _description_. Defaults to 24.
        frequency (str, optional): _description_. Defaults to 'H'.
        level (list, optional): _description_. Defaults to [90].
        ts_name_id (str, optional): _description_. Defaults to 'hourly_load'.

    Returns:
        pd.DataFrame: _description_
    """
    df_sf = prepare_statsforecast_df(data, ts_name_id)
    sf = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=frequency
            )
    sf.fit(df_sf)
    forecasts_df = sf.forecast(h=h_, level=level)
    forecasts_df_final = forecasts_df[["ds", "AutoARIMA"]]
    #return sf, forecasts_df_final
    if test is not None:
        plt.scatter(test.index, test, label="Observado")
    plt.title("AutoARIMA")
    plt.legend()
    plt.savefig(os.path.join(FORECASTS_FIG_DIR, "autoarima.png"))
    #plt.show()
    if write:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        #file_path = os.path.join(PATHS["forecasts_data"], f"holtwinters_fc_{now}.parquet")
        file_path = os.path.join(fcs_dir, f"autoatima_fc_{now}.parquet")
        forecasts_df_final.to_parquet(file_path)
    if save_model:
        joblib.dump(sf, '../models/autoarima_joblib')
    # sf.models[0].fit(df_sf["y"].values)
    # model_coefs = sf.models[0].model_
    # print(model_coefs)
    return forecasts_df_final