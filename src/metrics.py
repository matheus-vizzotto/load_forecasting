import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import     MeanAbsolutePercentageError
from math import sqrt
from typing import Optional

def get_metrics(forecast: pd.Series,
                test: pd.Series):
                #set_forecast_index: Optional[str]=None):
    forecast = forecast.values
    test = test.values
    #errors = [(test.iloc[i] - forecast.iloc[i]) for i in range(len(test))]
    errors = [(test[i] - forecast[i]) for i in range(len(test))]
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test, forecast)
    smape_obj = MeanAbsolutePercentageError(symmetric=True)
    smape = smape_obj(test, forecast)
    r2 = r2_score(test, forecast)
    measures = {"erro": sum(errors),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "smape": smape,
                "r2": r2
            }
    return measures