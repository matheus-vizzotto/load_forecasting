import pandas as pd
from utils import data_wrangling as dw
from utils.ts_wrangling import SerieTemporal
from models_ import Projecoes, ts_cross_validation
from models_ import (
    prophet_model,
    holtwinters_model
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# IDENTIFICADORES
INIT = "2012-01-01"
END = "2023-04-30"
#END = None
PERIOD = 24*365
HORIZON = 24*2

load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
if END:
    df_load = dw.pipeline(load).loc[INIT:END,:]
elif END is None:
    df_load = dw.pipeline(load).loc[INIT:,:]
df_load = df_load.iloc[-PERIOD:,:]
ts = SerieTemporal(data=df_load, y_col = "load_mwmed", date_col_name = "date", test_size=HORIZON, frequency='h')

# CROSS VALIDATION
INITIAL = "180 days"
CV_HORIZON = f"{HORIZON} hours"

def run_cross_validation():
    dataset = ts.full_series
    dataset = dataset.reset_index().set_index("date")
    cv = ts_cross_validation(data = dataset)
    cv.slider(use_x_days=INITIAL, horizon=CV_HORIZON,y_col="load_mwmed")
    models_evaluation = {}
    # Prophet
    models_evaluation["prophet"] = {}
    for partition in cv.partitions:
        d_train = cv.partitions[partition]["treino"]
        d_test = cv.partitions[partition]["teste"]
        df_train = pd.DataFrame(d_train, columns=["ds", "y"])
        df_test = pd.DataFrame(d_test, columns=["ds", "y"])
        print(partition)
        model = prophet_model(df_train)
        future_fbp = model.make_future_dataframe(periods=HORIZON, freq='h', include_history=False)
        forecast = model.predict(future_fbp)[["ds","yhat"]]
        df_comp = pd.merge(forecast, df_test, on="ds", how="right")
        metricas = get_metrics(df_comp["yhat"], df_comp["y"])
        #print(metricas["mape"])
        models_evaluation["prophet"][partition] = metricas
        print(models_evaluation)
