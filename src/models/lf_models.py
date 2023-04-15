import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utils.data_wrangling import prepare_statsforecast_df

# MODELOS
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    HoltWinters,
    CrostonClassic as Croston, 
    CrostonOptimized,
    TSB,
    MSTL,
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive,
    AutoETS,
    AutoCES,
    AutoTheta,
)
# MÉTRICAS
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sktime.performance_metrics.forecasting import     MeanAbsolutePercentageError

def compare_models(df: pd.DataFrame, 
                   h_: int,
                   season_lenght: int=24,
                   frequency: str='H', 
                   level=[90],
                   ts_name_id= 'hourly_load'):
    """_summary_
    """
    models = [
        AutoARIMA(season_length=24),
        #HoltWinters(), #erro: nonseasonal
        Croston(),
        CrostonOptimized(),
        SeasonalNaive(season_length=24),
        HistoricAverage(),
        DOT(season_length=24),
        #TSB(),
        MSTL(season_length=[24, 24*7]),
        HistoricAverage(),
        AutoETS(),
        AutoCES(),
        AutoTheta()
        ]   
    df_sf = prepare_statsforecast_df(df, ts_name_id)
    sf = StatsForecast(
        df=df_sf, 
        models=models,
        freq='H', 
        n_jobs=-1,
        fallback_model = SeasonalNaive(season_length=24)
        )
    sf.fit(df_sf)
    forecasts_df = sf.forecast(h=h_, level=level)
    #forecasts_df_final = forecasts_df[["ds", "AutoARIMA"]]
    return sf, forecasts_df


def auto_arima_model(df: pd.DataFrame, 
                     h_: int, 
                     season_length=24, 
                     frequency: str='H', 
                     level=[90],
                     ts_name_id= 'hourly_load') -> pd.DataFrame: 
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
    df_sf = prepare_statsforecast_df(df, ts_name_id)
    sf = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=frequency
    )
    sf.fit(df_sf)
    forecasts_df = sf.forecast(h=h_, level=level)
    forecasts_df_final = forecasts_df[["ds", "AutoARIMA"]]
    return forecasts_df_final

def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    for model in models:
        eval_ = df.groupby(['unique_id', 'cutoff']).apply(lambda x: metric(x['y'].values, x[model].values)).to_frame() # Calculate loss for every unique_id, model and cutoff.
        eval_.columns = [model]
        evals.append(eval_)
    evals = pd.concat(evals, axis=1)
    evals = evals.groupby(['unique_id']).mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals

def crossval(sf: StatsForecast, 
             data: pd.DataFrame, 
             h: int=24, 
             step_size: int=24, 
             windows: int=2):
    """_summary_

    Args:
        sf (StatsForecast): _description_
        data (pd.DataFrame): _description_
        h (int, optional): _description_. Defaults to 24.
        step_size (int, optional): _description_. Defaults to 24.
        windows (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    crossvalidation_df = sf.cross_validation(
        df=data,
        h=h,
        step_size=step_size,
        n_windows=windows
        )
    return crossvalidation_df

def crossval_summary_table(crossvalidation_df: pd.DataFrame, 
                           metrics: list=None) -> pd.DataFrame:
    """_summary_

    Args:
        crossvalidation_df (pd.DataFrame): _description_
        metrics (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    metrics = [
            mean_absolute_error,
            mean_squared_error,
            mean_squared_error,
            mean_absolute_percentage_error,
            r2_score,
            MeanAbsolutePercentageError(symmetric=True)
            ]
    metrics_df = pd.DataFrame()
    for metric in metrics:
        evaluation_df = evaluate_cross_validation(crossvalidation_df, metric)
        try:
            evaluation_df["metric"] = metric.__name__
        except:
            evaluation_df["metric"] = type(metric).__name__
        metrics_df = pd.concat([metrics_df, evaluation_df])
    summary_df = metrics_df.groupby('best_model').size().sort_values().to_frame()
    #summary_df.reset_index().columns = ["Model", "Nr. of unique_ids"]
    return metrics_df, summary_df