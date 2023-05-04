# TODO: TRAZER RESTANTE DOS MODELOS PARA O MÓDULO Projecoes.

import fbprophet
from fbprophet.diagnostics import cross_validation
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from statsforecast.models import MSTL
from utils.data_wrangling import prepare_statsforecast_df
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from datetime import datetime
import joblib
from paths import PATHS
import os

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
        self.data = data
        self.y_col = y_col
        self.date_col_name = date_col_name
        self.train = data.iloc[:-test_size][y_col]
        self.horizon = test_size
        self.test = data.iloc[-test_size:][y_col]
        self.frequency = frequency  
        self.seasonality = seasonality

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

    def plot_forecasting(self, 
                         yhat: pd.Series,
                         plot_name: str):
        """_summary_

        Args:
            yhat (pd.Series): _description_
            plot_name (str): _description_
        """
        plt.figure(figsize=(15,5))
        plt.plot(self.ts.test.index, yhat, label='Forecast')
        plt.scatter(self.ts.test.index, self.ts.test, label='Observado')
        plt.title(plot_name)
        plt.legend()
        plt.savefig(os.path.join(self.forecasts_fig_dir, f"{plot_name}.png"))
    
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
        model = fbprophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        self.models[model_name] = model
        future_fbp = model.make_future_dataframe(periods=self.ts.horizon, freq=self.ts.frequency, include_history=False)
        forecast = model.predict(future_fbp)
        self.forecasts[model_name] = forecast
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
        model = ExponentialSmoothing(series, seasonal_periods=self.ts.seasonality, trend=trend, seasonal=seasonal, freq=self.ts.frequency).fit()
        self.models[model_name] = model
        forecast = model.forecast(self.ts.horizon)
        self.plot_forecasting(yhat=forecast, plot_name=f"{model_name}")
        forecast_df = pd.DataFrame({"date": forecast.index.values, "load_mwmed": forecast.values})
        self.forecasts[model_name] = forecast_df
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast_df.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(model, model_path)
        return forecast
    
    def auto_arima_model(self,
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
        forecast = sf.forecast(h=self.ts.horizon, level=level)
        self.forecasts[model_name] = forecast
        self.plot_forecasting(yhat=forecast[model_name], plot_name=f"{model_name}")
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return forecast
    
    def mstl_model(self,
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
        forecast = sf.forecast(h=self.ts.horizon, level=level)
        self.forecasts[model_name] = forecast
        self.plot_forecasting(yhat=forecast[model_name], plot_name=f"{model_name}")
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.forecasts_dir, f"{model_name}_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            model_path = os.path.join(self.models_dir, f'{model_name}_joblib')
            joblib.dump(sf, model_path)
        return forecast
        