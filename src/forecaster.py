from models.prophet_model import prophet_model 
from models.holtwinters_model import holt_winters_model
from utils import data_wrangling as dw
from utils import ts_wrangling as tw
import os

# script_dir = os.path.dirname(os.path.realpath(__file__))
# os.chdir(script_dir)

INIT = "2012-01-01"
END = "2023-02-28"
PERIOD = 24*365
HORIZON = 24*2

load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
df_load = dw.pipeline(load).loc[INIT:END,:]
df_load = df_load.iloc[-PERIOD:,:]
train, test = tw.train_test_split(df_load, test=HORIZON)



def run_models():
    fc_p = prophet_model(data=train, horizon=HORIZON, test=test["load_mwmed"], save_model=True)
    fc_hw = holt_winters_model(data=train, y_col="load_mwmed", horizon=HORIZON, seasonality=24, trend_="add", seasonal_="mul", test=test["load_mwmed"], save_model=True)


# COLOCAR RESTANTE DOS MODELOS