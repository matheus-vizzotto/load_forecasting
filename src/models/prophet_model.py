import fbprophet
from fbprophet.diagnostics import cross_validation
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime
import joblib

def prepare_prophet_df(data: pd.DataFrame, 
                       cols: List[str],
                       date_in_index: bool=True) -> pd.DataFrame:
    if date_in_index:
        ts = data.reset_index()
    else:
        ts = data.copy()
    ts = ts[cols]
    ts.columns = ["ds", "y"]
    return ts

def prophet_model(data: pd.DataFrame, 
                  #date_col: str, 
                  #y_col: str, 
                  horizon: int, 
                  test: pd.Series=None,
                  frequency: str='h',
                  write: bool=True,
                  save_model=False):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        date_col (str): _description_
        y_col (str): _description_
        horizon (int): _description_
        test (pd.DataFrame, optional): _description_. Defaults to None.
        frequency (str, optional): _description_. Defaults to 'h'.

    Returns:
        _type_: _description_
    """
    x = prepare_prophet_df(data, ["date", "load_mwmed"])
    # df = x[[date_col,y_col]]
    # df.rename(columns={
    #             date_col: 'ds', 
    #             y_col: 'y'
    #             }, inplace=True)
    model = fbprophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(x)
    future_fbp = model.make_future_dataframe(periods=horizon, freq=frequency, include_history=False)
    forecast = model.predict(future_fbp)
    plt.figure(figsize=(15,5))
    plt.plot(forecast['ds'], forecast.yhat, label="Forecast")
    plt.fill_between(forecast['ds'], forecast['yhat_upper'], forecast['yhat_lower'], alpha=.2, color="darkblue")
    if test is not None:
        plt.scatter(test.index, test, label="Observado")
    plt.legend()
    plt.show()
    if write:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        forecast.to_parquet(f"../data/forecasts/prophet_fc_{now}.parquet")
    if save_model:
        joblib.dump(model, '../models/prophet_joblib')
    return forecast