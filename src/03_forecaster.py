from models.prophet_model import prophet_model 
from utils import data_wrangling as dw
from utils import ts_wrangling as tw
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

INIT = "2012-01-01"
END = "2023-02-28"
PERIOD = 24*365
HORIZON = 24*2

load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
df_load = dw.pipeline(load).loc[INIT:END,:]
df_load = df_load.iloc[-PERIOD:,:]
train, test = tw.train_test_split(df_load, test=HORIZON)


fc = prophet_model(data=train, horizon=HORIZON, test=test["load_mwmed"])


# COLOCAR RESTANTE DOS MODELOS