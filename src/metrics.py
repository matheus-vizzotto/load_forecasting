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

def get_cumulative_metrics(X: pd.DataFrame,
                           date_col: str,
                           yhat: str,
                           y: str,
                           model_name: Optional[str]) -> pd.DataFrame:
    """_summary_

    Args:
        X (pd.DataFrame): _description_
        date_col (str): _description_
        yhat (str): _description_
        y (str): _description_
        model_name (Optional[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    X = X.copy().sort_values(by=date_col)
    cum_metrics = []
    for i in range(1, len(X)+1):
        #print(df_model_sub["datetime"].iloc[i])
        part_df = X.loc[:i, [yhat, y]]
        metrics = {}
        if model_name:
            metrics["model"] = model_name
        metrics["i"] = i
        metrics.update(get_metrics(part_df["yhat"], part_df["y"]))
        cum_metrics.append(metrics)
    df_cum_metrics = pd.DataFrame(cum_metrics)
    return df_cum_metrics

def get_best_model(X: pd.DataFrame):
    """Recebe o DataFrame gerado por run_e valuation (get_cumulative_metrics
    para todos os modelos e retorna o modelo com melhor desempenho geral no 
    final do horizonte de previsão.

    Args:
        X (pd.DataFrame): _description_
    """
    df_horizon = X[X["i"]==X["i"].max()].set_index("model").T
    df_horizon["best_model"] = df_horizon.idxmin(axis=1)
    best_model = df_horizon.groupby("best_model").size().sort_values().to_frame().iloc[-1].name
    return best_model