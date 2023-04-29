import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from datetime import datetime
import joblib
from typing import Optional
# from paths import PATHS
import os

# import warnings # retirar avisos
# warnings.filterwarnings('ignore')
# rcParams['figure.figsize'] = 15, 5


# df = load_data()
# df["load_mwmed"].interpolate(method = "linear", inplace = True)
# n_test = 15
# train, test = train_test_split(df, n_test)
#fit1 = ExponentialSmoothing(train ,seasonal_periods=7,trend='add', seasonal='mul').fit() # seasonal='mul' é melhor
# fit1 = ExponentialSmoothing(train ,seasonal_periods=7,trend='add', seasonal='add').fit() # seasonal='mul' é melhor
# y_hat = fit1.forecast(n_test)


def holt_winters_model(data: pd.DataFrame,
                       #y_col: str,
                       horizon: int,
                       seasonality: int,
                       trend_: str,
                       seasonal_: str,
                       test: Optional[pd.Series]=None,
                       write: bool=True,
                       fcs_dir: Optional[str]=None,
                       save_model: bool=False) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        y_col (str): _description_
        horizon (int): _description_
        seasonality (int): _description_
        trend_ (str): _description_
        seasonal_ (str): _description_
        test (Optional[pd.Series]): _description_
        write (bool, optional): _description_. Defaults to True.
        save_model (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    #x = data[y]
    fitted_model = ExponentialSmoothing(data, seasonal_periods=seasonality, trend=trend_, seasonal=seasonal_).fit()
    forecast = fitted_model.forecast(horizon)
    forecast_df = pd.DataFrame({"date": forecast.index.values, "load_mwmed": forecast.values})
    plt.figure(figsize=(15,5))
    plt.plot(forecast_df["date"], forecast_df["load_mwmed"], label="Forecast")
    if test is not None:
        plt.scatter(test.index, test, label="Observado")
    if write:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        #file_path = os.path.join(PATHS["forecasts_data"], f"holtwinters_fc_{now}.parquet")
        file_path = os.path.join(fcs_dir, f"holtwinters_fc_{now}.parquet")
        forecast_df.to_parquet(file_path)
    if save_model:
        joblib.dump(fitted_model, '../models/holtwinters_joblib')
    return forecast

    
