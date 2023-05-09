import fbprophet
from fbprophet.diagnostics import cross_validation
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
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
warnings.simplefilter('ignore', ConvergenceWarning)

PROCESSED_DATA_DIR = PATHS['processed_data']
FCS_PATH = PATHS["forecasts_data"]
FORECASTS_FIG_DIR = PATHS['forecasts_figs']
MODELS_DIR = PATHS['models']

class SerieTemporal:
    """Classe para armazenar informações da série temporal.
    frequency: str
        Frequência da série temporal. Pode ser 'H' (horária) ou 'd' (diária).
    """
    def __init__(self,
                 data: pd.DataFrame,
                 y_col: str,
                 date_col_name: str,
                 test_size: int,
                 frequency: str,
                 seasonality: int=24):
        """
        Args:
            data (pd.DataFrame): DataFrame onde o índice é a data-hora.
            y_col (str): Nome da coluna com a variável-alvo.
            date_col_name (str): Nome da coluna (índice) com a data-hora.
            test_size (int): Tamanho da partição de teste.
            frequency (str): Frequência da série. "H" ou "d".
            seasonality (int, optional): Intervalo de tempo cíclico. Padrão de 24 (horas).
        """
        self.data = data
        self.y_col = y_col
        self.full_series = data[y_col]
        self.date_col_name = date_col_name
        self.train = data.iloc[:-test_size][y_col]
        self.horizon = test_size
        self.test = data.iloc[-test_size:][y_col]
        self.frequency = frequency  
        self.seasonality = seasonality
        self.seasonal_components = get_seasonal_components(data.index)

class Projecoes:
    def __init__(self, 
                 ts: SerieTemporal):
        self.ts = ts
        self.ts_data = self.ts.train.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        self.models = {}
        self.forecasts = {}
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
        future_fbp = model.make_future_dataframe(periods=self.ts.horizon, freq=self.ts.frequency, include_history=False)
        forecast = model.predict(future_fbp)
        self.forecasts[model_name] = forecast
        metrics = get_metrics(forecast["yhat"], self.ts.test)
        self.models_metrics[model_name] = metrics 
        self.plot_forecasting(yhat=forecast["yhat"], plot_name=model_name)
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(model, model_path)
        return forecast
    
    def hw_fit_forecast(self, 
                        trend: str='add',
                        seasonal: str='add',
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
        model = holtwinters_model(series, seasonal_periods=self.ts.seasonality, trend=trend, seasonal=seasonal, freq=self.ts.frequency)
        self.models[model_name] = model
        forecast = model.forecast(self.ts.horizon)
        self.plot_forecasting(yhat=forecast, plot_name=f"{model_name}")
        forecast_df = pd.DataFrame({"date": forecast.index.values, "load_mwmed": forecast.values})
        self.forecasts[model_name] = forecast_df
        metrics = get_metrics(forecast, self.ts.test)
        self.models_metrics[model_name] = metrics 
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast_df.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(model, model_path)
        return forecast
    
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
        forecast = sf.forecast(h=self.ts.horizon, level=level)
        self.forecasts[model_name] = forecast
        metrics = get_metrics(forecast[model_name], self.ts.test)
        self.models_metrics[model_name] = metrics 
        self.plot_forecasting(yhat=forecast[model_name], plot_name=f"{model_name}")
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return forecast
    
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
        forecast = sf.forecast(h=self.ts.horizon, level=level)
        self.forecasts[model_name] = forecast
        metrics = get_metrics(forecast[model_name], self.ts.test)
        self.models_metrics[model_name] = metrics 
        self.plot_forecasting(yhat=forecast[model_name], plot_name=f"{model_name}")
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return forecast
        
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
            t[i]["treino"] = series_to_tuples(index=x.index.astype('str'), y=x.loc[:,y_col])
            test = self.data.loc[end+pd.Timedelta("1 hour"):end+pd.Timedelta("49 hours")]
            t[i]["teste"] = series_to_tuples(index=test.index.astype('str'), y=test.loc[:,y_col])
        self.partitions = t

    def store_partitions(self):
        if self.partitions:
            file_path = os.path.join(PROCESSED_DATA_DIR, "cross_validation_partitions.json")
            with open(file_path, 'w') as f:
                json.dump(self.partitions, f)
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

def holtwinters_model(data, seasonal_periods, trend, seasonal, freq):
    model = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, freq=freq)
    model = model.fit()
    return model

def autoarima_model():
    pass

def mstl_model():
    pass

def series_to_tuples(index: pd.DatetimeIndex, 
                     y: pd.Series):
    if not isinstance(y, pd.Series):
        raise Exception("Os valores não têm o tipo de Pandas Series. Tente selecionar a coluna de interesse.")
    l_tuples = list(zip(index, y.values))
    return l_tuples