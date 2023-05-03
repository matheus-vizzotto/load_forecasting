# TODO: TRAZER RESTANTE DOS MODELOS PARA O MÓDULO Projecoes.

import fbprophet
from fbprophet.diagnostics import cross_validation
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from statsforecast.models import MSTL
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
                 frequency: str):
        self.data = data
        self.y_col = y_col
        self.date_col_name = date_col_name
        self.train = data.iloc[:-test_size][y_col]
        self.horizon = test_size
        self.test = data.iloc[-test_size:][y_col]
        self.frequency = frequency  

class Projecoes:
    def __init__(self, 
                 ts: SerieTemporal,
                 fcs_dir: Optional[str]=FCS_PATH):
        self.ts = ts
        self.models = {}
        self.forecasts = {}

    def plot_forecasting(self, 
                         yhat: pd.Series,
                         plot_name: str):
        plt.figure(figsize=(15,5))
        plt.plot(self.ts.test.index, yhat, label='Forecast')
        print(self.ts.test)
        plt.scatter(self.ts.test.index, self.ts.test, label='Observado')
        plt.title(plot_name)
        plt.legend()
        plt.savefig(os.path.join(FORECASTS_FIG_DIR, f"{plot_name}.png"))
    
    def prophet_fit_forecast(self, 
                             write: bool=True,
                             save_model: bool=True):
        df = self.ts.train.reset_index()[[self.ts.date_col_name, self.ts.y_col]].copy()
        df.columns = ["ds", "y"]
        model = fbprophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        self.models['prophet'] = model
        future_fbp = model.make_future_dataframe(periods=self.ts.horizon, freq=self.ts.frequency, include_history=False)
        forecast = model.predict(future_fbp)
        self.forecasts['prophet'] = forecast
        self.plot_forecasting(yhat=forecast["yhat"], plot_name="prophet2")
        if write:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(FCS_PATH, f"prophet_fc_{now}.parquet")
            forecast.to_parquet(file_path)
        if save_model:
            #joblib.dump(model, '../models/prophet_joblib')
            model_path = os.path.join(MODELS_DIR, 'prophet_joblib')
            joblib.dump(model, model_path)
        return forecast
