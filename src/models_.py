from utils.ts_wrangling import SerieTemporal
import fbprophet
from fbprophet.diagnostics import cross_validation
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
import numpy as np
from statsforecast.models import MSTL
from utils.data_wrangling import prepare_statsforecast_df, get_seasonal_components
from metrics import get_metrics
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import json
from datetime import datetime
import joblib
from paths import PATHS
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import copy
warnings.simplefilter('ignore', ConvergenceWarning)

PROCESSED_DATA_DIR = PATHS['processed_data']
FCS_PATH = PATHS["forecasts_data"]
FORECASTS_FIG_DIR = PATHS['forecasts_figs']
MODELS_DIR = PATHS['models']

class Projecoes:
    def __init__(self, 
                 ts: SerieTemporal):
        self.ts = ts
        self.ts_data = self.ts.train.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        self.models = {}
        self.is_forecasts = {}
        self.oos_forecasts = {}
        self.forecasts_dir = FCS_PATH
        self.forecasts_fig_dir = FORECASTS_FIG_DIR
        self.models_dir = MODELS_DIR
        self.models_metrics = {}

    def plot_forecasting(self, 
                         yhat: pd.Series,
                         plot_name: str):
        """_summary_

        Args:
            yhat (pd.Series): _description_
            plot_name (str): _description_
        """
        plt.figure(figsize=(15,5))
        plt.plot(self.ts.test.index, yhat, label='Forecast', color="red")
        plt.scatter(self.ts.test.index, self.ts.test, label='Observado')
        plt.title(plot_name)
        plt.legend()
        plt.savefig(os.path.join(self.forecasts_fig_dir, f"{plot_name}.png"))
        plt.close()
    
    def prophet_fit_forecast(self, 
                             write: bool=True,
                             save_model: bool=True,
                             model_name: str="Prophet"):
        """_summary_

        Args:
            write (bool, optional): _description_. Defaults to True.
            save_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        df = self.ts.train.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        df.columns = ["ds", "y"]
        # model = fbprophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        # model.fit(df)
        model = prophet_model(df)
        self.models[model_name] = model
        future_fbp = model.make_future_dataframe(periods=self.ts.horizon, freq=self.ts.frequency, include_history=True)
        forecast = model.predict(future_fbp)[["ds", "yhat"]].set_index("ds")
        is_forecasts, oos_forecasts = forecast.iloc[:-self.ts.horizon], forecast.iloc[-self.ts.horizon:] # is = in_sample; oos = out_of_sample
        self.is_forecasts[model_name] = self.oos_forecasts[model_name] = self.models_metrics[model_name] = self.models_metrics[model_name]["oos"] = self.models_metrics[model_name]["is"] = {}
        self.is_forecasts[model_name] = is_forecasts
        self.oos_forecasts[model_name] = oos_forecasts
        self.models_metrics[model_name]["oos"] = get_metrics(oos_forecasts, self.ts.test)
        self.models_metrics[model_name]["is"] = get_metrics(is_forecasts, self.ts.train)
        self.plot_forecasting(oos_forecasts, plot_name=f"oos_{model_name}")

        df = self.ts.full_series.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        df.columns = ["ds", "y"]
        model = prophet_model(df)
        self.models[model_name] = model
        future_fbp = model.make_future_dataframe(periods=self.ts.horizon, freq=self.ts.frequency, include_history=False)
        forecast = model.predict(future_fbp).set_index("ds").loc[:,"yhat"]
        if write:
            # OUT-OF-SAMPLE
            #now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"oos_{model_name}_fc.parquet")
            oos_forecasts.reset_index().to_parquet(file_path)
            # IN-SAMPLE
            file_path = os.path.join(self.forecasts_dir, f"is_{model_name}_fc.parquet")
            is_forecasts.reset_index().to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f"{model_name}_joblib")
            joblib.dump(model, model_path)
        return [model_name, generate_forecast_df(forecast.index, forecast.values)]
    
    def hw_fit_forecast(self, 
                        trend: str='add',
                        seasonal: str='add',
                        damped_trend: bool=False,
                        write: bool=True,
                        save_model: bool=True,
                        model_name: str="HoltWinters"):
        """_summary_

        Args:
            write (bool, optional): _description_. Defaults to True.
            save_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        #df = self.ts.train.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        series = self.ts.train
        #model = ExponentialSmoothing(series, seasonal_periods=self.ts.seasonality, trend=trend, seasonal=seasonal, freq=self.ts.frequency).fit()
        model = holtwinters_model(series, seasonal_periods=self.ts.seasonality, trend=trend, damped_trend=damped_trend, seasonal=seasonal, freq=self.ts.frequency)
        self.models[model_name] = model
        oos_forecast = model.forecast(self.ts.horizon)
        is_forecast = model.predict(start=0, end=len(series)-1)
        self.plot_forecasting(yhat=oos_forecast, plot_name=f"oos_{model_name}")
        forecast_df = pd.DataFrame({"date": oos_forecast.index.values, "load_mwmed": oos_forecast.values})
        self.is_forecasts[model_name] = self.oos_forecasts[model_name] = self.models_metrics[model_name] = self.models_metrics[model_name]["oos"] = self.models_metrics[model_name]["is"] = {}
        self.is_forecasts[model_name] = is_forecast
        self.oos_forecasts[model_name] = oos_forecast
        self.models_metrics[model_name]["oos"] = get_metrics(oos_forecast, self.ts.test)
        self.models_metrics[model_name]["is"] = get_metrics(is_forecast, self.ts.train)
        #self.plot_forecasting(oos_forecast, plot_name=f"oos_{model_name}")
        series = self.ts.full_series.copy()
        model = holtwinters_model(series, seasonal_periods=self.ts.seasonality, trend=trend, damped_trend=damped_trend, seasonal=seasonal, freq=self.ts.frequency)
        self.models[model_name] = model
        forecast = model.forecast(self.ts.horizon)
        if write:
            # OUT-OF-SAMPLE
            #now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"oos_{model_name}_fc.parquet")
            oos_forecast = oos_forecast.reset_index()
            oos_forecast.columns = ["date", "yhat"]
            oos_forecast.reset_index().to_parquet(file_path)
            # IN-SAMPLE
            file_path = os.path.join(self.forecasts_dir, f"is_{model_name}_fc.parquet")
            is_forecast = is_forecast.reset_index()
            is_forecast.columns = ["date", "yhat"]
            is_forecast.reset_index().to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f"{model_name}_joblib")
            joblib.dump(model, model_path)
        return [model_name, generate_forecast_df(forecast.index, forecast.values)]
    
    def auto_arima_fit_forecast(self,
                                level: List[int]=[90],
                                ts_name_id= 'hourly_load',
                                write: bool=True,
                                save_model: bool=True,
                                model_name: str="AutoARIMA") -> pd.DataFrame:
        """_summary_

        Args:
            level (list, optional): _description_. Defaults to [90].
            ts_name_id (str, optional): _description_. Defaults to 'hourly_load'.
            write (bool, optional): _description_. Defaults to True.
            save_model (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        df_sf = prepare_statsforecast_df(self.ts.train, ts_name_id)
        sf = StatsForecast(
            models=[AutoARIMA(season_length=self.ts.seasonality)],
            freq=self.ts.frequency
            )
        sf.fit(df_sf)
        self.models[model_name] = sf
        oos_forecast = sf.forecast(h=self.ts.horizon, level=level, fitted=True)[["ds", model_name]].set_index("ds")
        is_forecast = sf.forecast_fitted_values()[["ds", model_name]].set_index("ds")
        self.plot_forecasting(yhat=oos_forecast[model_name], plot_name=f"oos_{model_name}")
        self.is_forecasts[model_name] = self.oos_forecasts[model_name] = self.models_metrics[model_name] = self.models_metrics[model_name]["oos"] = self.models_metrics[model_name]["is"] = {}
        self.is_forecasts[model_name] = is_forecast
        self.oos_forecasts[model_name] = oos_forecast
        self.models_metrics[model_name]["oos"] = get_metrics(oos_forecast, self.ts.test)
        self.models_metrics[model_name]["is"] = get_metrics(is_forecast, self.ts.train)

        df_sf = prepare_statsforecast_df(self.ts.full_series, ts_name_id)
        sf = StatsForecast(
            models=[AutoARIMA(season_length=self.ts.seasonality)],
            freq=self.ts.frequency
            )
        sf.fit(df_sf)
        # obter parâmetros
        # sf.models[0].fit(df_sf.values)
        # sf.models[0].model_["arma"]
        self.models[model_name] = sf
        forecast = sf.forecast(h=self.ts.horizon, level=level).set_index("ds").loc[:,model_name]
        if write:
             # OUT-OF-SAMPLE
             #now = datetime.now().strftime("%Y%m%d_%H%M%S")
             file_path = os.path.join(self.forecasts_dir, f"oos_{model_name}_fc.parquet")
             oos_forecast = oos_forecast.reset_index()
             oos_forecast.columns = ["date", "yhat"]
             oos_forecast.reset_index().to_parquet(file_path)
             # IN-SAMPLE
             file_path = os.path.join(self.forecasts_dir, f"is_{model_name}_fc.parquet")
             is_forecast = is_forecast.reset_index()
             is_forecast.columns = ["date", "yhat"]
             is_forecast.reset_index().to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return [model_name, generate_forecast_df(forecast.index, forecast.values)]
    
    def mstl_fit_forecast(self,
                   level=[90],
                   ts_name_id= 'hourly_load',
                   write: bool=True,
                   save_model: bool=True,
                   model_name: str="MSTL") -> pd.DataFrame:
        """_summary_

        Args:
            level (list, optional): _description_. Defaults to [90].
            ts_name_id (str, optional): _description_. Defaults to 'hourly_load'.
            write (bool, optional): _description_. Defaults to True.
            save_model (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        df_sf = prepare_statsforecast_df(self.ts.train, ts_name_id)
        sf = StatsForecast(
            models=[MSTL(season_length=[self.ts.seasonality, self.ts.seasonality*7])],
            freq=self.ts.frequency
            )
        sf.fit(df_sf)
        self.models[model_name] = sf
        oos_forecast = sf.forecast(h=self.ts.horizon, level=level, fitted=True)[["ds", model_name]].set_index("ds")
        is_forecast = sf.forecast_fitted_values()[["ds", model_name]].set_index("ds")
        self.plot_forecasting(yhat=oos_forecast[model_name], plot_name=f"oos_{model_name}")
        self.is_forecasts[model_name] = self.oos_forecasts[model_name] = self.models_metrics[model_name] = self.models_metrics[model_name]["oos"] = self.models_metrics[model_name]["is"] = {}
        self.is_forecasts[model_name] = is_forecast
        self.oos_forecasts[model_name] = oos_forecast
        self.models_metrics[model_name]["oos"] = get_metrics(oos_forecast, self.ts.test)
        self.models_metrics[model_name]["is"] = get_metrics(is_forecast, self.ts.train)

        df_sf = prepare_statsforecast_df(self.ts.full_series, ts_name_id)
        sf = StatsForecast(
            models=[MSTL(season_length=[self.ts.seasonality, self.ts.seasonality*7])],
            freq=self.ts.frequency
            )
        sf.fit(df_sf)
        self.models[model_name] = sf
        forecast = sf.forecast(h=self.ts.horizon, level=level).set_index("ds").loc[:,model_name]
        
        if write:
            # OUT-OF-SAMPLE
            #now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"oos_{model_name}_fc.parquet")
            oos_forecast = oos_forecast.reset_index()
            oos_forecast.columns = ["date", "yhat"]
            oos_forecast.reset_index().to_parquet(file_path)
            # IN-SAMPLE
            file_path = os.path.join(self.forecasts_dir, f"is_{model_name}_fc.parquet")
            is_forecast = is_forecast.reset_index()
            is_forecast.columns = ["date", "yhat"]
            is_forecast.reset_index().to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return [model_name, generate_forecast_df(forecast.index, forecast.values)]
        
class ts_cross_validation:
    def __init__(self, data):
        #self.ts = ts
        self.data = data
        self.cutoff_dates = []
        self.partitions = []
        #self.models = models or []

    def slider(self,
               use_x_days: str,
               horizon: int,
               y_col: str):
        # if horizon is None:
        #     horizon = pd.Timedelta(f"{self.ts.horizon} hours")
        use_x_days = pd.Timedelta(use_x_days)
        horizon = pd.Timedelta(horizon)
        min_date = self.data.index.min()
        max_date = self.data.index.max()
        initial_training_date = min_date + use_x_days
        n_partitions = int(len(pd.date_range(initial_training_date, max_date, freq='h'))/(horizon.total_seconds()/3600))
        t = {}
        for i in range(1, n_partitions+1):
            start = self.data.index.max() - ((horizon*(i+1)) - pd.Timedelta("1 hour"))
            self.cutoff_dates.append(start)
            end = start + (horizon - pd.Timedelta("1 hour"))
            x = self.data.loc[:end]#.index
            #self.partitions.append(self.data.loc[x])
            t[i] = {}
            t[i]["treino"] = series_to_tuples(index=x.index, y=x.loc[:,y_col])
            test = self.data.loc[end+pd.Timedelta("1 hour"):end+pd.Timedelta("48 hours")]
            t[i]["teste"] = series_to_tuples(index=test.index, y=test.loc[:,y_col])
        self.partitions = t

    def store_partitions(self):
        partitions_dict = copy.deepcopy(self.partitions)
        if partitions_dict:
            for partition in range(1, len(partitions_dict)+1):
                partitions_dict[partition]["treino"] = [(str(x[0]),x[1]) for x in partitions_dict[partition]["treino"]]
                partitions_dict[partition]["teste"] = [(str(x[0]),x[1]) for x in partitions_dict[partition]["teste"]]
            file_path = os.path.join(PROCESSED_DATA_DIR, "cross_validation_partitions.json")
            with open(file_path, 'w') as f:
                json.dump(partitions_dict, f)
        else:
            raise Exception("Atributo 'partitions' vazio: rode o cross-validation antes de salvar.")
        

            
        

    # def add_model(self, model):
    #     self.models.append(model)

    # def run_validation_models(self):
    #     for model in self.models:
    #         for partition in self.training_dates:
    #             data = self.ts.train.loc[self.training_dates[partition]]
    #             ts = SerieTemporal(data=data, y_col = "load_mwmed", date_col_name = "date", test_size=HORIZON, frequency='h')
    #             fm = Projecoes(ts=ts)
    #             getattr(fm, model.__name__)

def prophet_model(data):
    """Fits a Prophet model that is used in Projecoes and in cross validation. 

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = fbprophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(data)
    return model

def holtwinters_model(data, seasonal_periods, trend, damped_trend, seasonal, freq):
    model = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, trend=trend, damped_trend=damped_trend, seasonal=seasonal, freq=freq)
    model = model.fit()
    return model

def autoarima_model():
    pass

def mstl_model():
    pass

def series_to_tuples(index: pd.DatetimeIndex, 
                     y: pd.Series):
    """Converte uma pandas.Series em lista de tuplas para armazenamento em json.

    Args:
        index (pd.DatetimeIndex): _description_
        y (pd.Series): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(y, pd.Series):
        raise Exception("Os valores não têm o tipo de Pandas Series. Tente selecionar a coluna de interesse.")
    l_tuples = list(zip(index, y.values))
    return l_tuples

def generate_forecast_df(date: pd.DatetimeIndex, values: np.ndarray):
    """Unifica a estrutura de armazenamento das projeções geradas a partir de uma pd.Series.
    Atenção: a forma de tratamento de dados anterior pode fazer com que a "série", de apenas
    uma coluna, seja na verdade um DataFrame; neste caso, tente utilizar x.set_index(<date>).loc[:<values>]
    com os respectivos nomes das colunas.

    Args:
        index (list): _description_
        values (list): _description_
    """
    col_names = ['datetime', 'yhat']
    df = pd.DataFrame({col_names[0]: date, col_names[1]: values})
    return df
