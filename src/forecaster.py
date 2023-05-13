import pandas as pd
from utils import data_wrangling as dw
from utils import ts_wrangling as tw
from utils.logger import timer_decorator
from models_ import SerieTemporal, Projecoes
from models_ import (
    prophet_model,
    holtwinters_model
)
from metrics import get_metrics
import joblib
from paths import PATHS
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
FCS_PATH = PATHS["forecasts_data"]
MODELS_DIR = PATHS['models']

# IDENTIFICADORES
INIT = "2012-01-01"
#END = "2023-04-30"
END = None
PERIOD = 24*365
HORIZON = 24*14

# DADOS
load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
if END:
    df_load = dw.pipeline(load).loc[INIT:END,:]
elif END is None:
    df_load = dw.pipeline(load).loc[INIT:,:]
df_load = df_load.iloc[-PERIOD:,:]
ts = SerieTemporal(data=df_load, y_col = "load_mwmed", date_col_name = "date", test_size=HORIZON, frequency='h')


fm = Projecoes(ts=ts)

# MODELOS
@timer_decorator
def run_models():
    print(f"\tPeríodo: {df_load.index.min()} a {df_load.index.max()}")
    print("RODANDO MODELOS")
    print("\t#### Prophet ####")
    fc_p = fm.prophet_fit_forecast()
    print(fm.models_metrics["Prophet"])
    print("\t#### Holt-Winters ####")
    fc_hw = fm.hw_fit_forecast()
    print(fm.models_metrics["HoltWinters"])
    print("\t#### MSTL ####")
    fc_ml = fm.mstl_fit_forecast()
    print(fm.models_metrics["MSTL"])
    # print("\t#### AutoARIMA ####")
    # fc_aa = fm.auto_arima_fit_forecast()
    # print(fm.models_metrics["AutoARIMA"])
    # model_path = os.path.join(MODELS_DIR, "AllModels_joblib")
    # joblib.dump(fm, model_path)
    return [fc_p, fc_hw, fc_ml]
    