from models.prophet_model import prophet_model 
from models.holtwinters_model import holt_winters_model
from models.arima_model import auto_arima_model
from models.mstl_model import mstl_model
from utils import data_wrangling as dw
from utils import ts_wrangling as tw
from utils.logger import timer_decorator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# IDENTIFICADORES
INIT = "2012-01-01"
END = "2023-02-28"
PERIOD = 24*365
HORIZON = 24*2

# NOVO SCRIPT "models_.py" (junta todos que estão no "models/")
from models_ import SerieTemporal, Projecoes
load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
df_load = dw.pipeline(load).loc[INIT:END,:]
df_load = df_load.iloc[-PERIOD:,:]
ts = SerieTemporal(data=df_load, y_col = "load_mwmed", date_col_name = "date", test_size=HORIZON, frequency='h')
fm = Projecoes(ts=ts)
#fm.prophet_fit_forecast()
#fm.hw_fit_forecast()
#fm.auto_arima_model()
#fm.mstl_model()

# # DATA PREP
# load = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
# df_load = dw.pipeline(load).loc[INIT:END,:]
# df_load = df_load.iloc[-PERIOD:,:]
# train, test = tw.train_test_split(df_load, test=HORIZON)
# Y_train, Y_test = train["load_mwmed"], test["load_mwmed"]


# MODELS/...
# @timer_decorator
#def run_models(fcs_dir_):
#     print("RODANDO MODELOS")
#     print("\t#### Prophet ####")
#     fc_p = prophet_model(data=Y_train, horizon=HORIZON, test=test["load_mwmed"], save_model=True, fcs_dir=fcs_dir_)
#     print("\t#### Holt-Winters ####")
#     fc_hw = holt_winters_model(data=Y_train, horizon=HORIZON, seasonality=24, trend_="add", seasonal_="mul", 
#                                test=test["load_mwmed"], save_model=True, fcs_dir=fcs_dir_)
#     print("\t#### AutoARIMA ####")
#     fc_aa = auto_arima_model(train, h_=HORIZON, level = [99,95,90], fcs_dir=fcs_dir_, test=test["load_mwmed"])
#     print("\t#### MSTL ####")
#     fc_aa = mstl_model(train, h_=HORIZON, level = [99,95,90], fcs_dir=fcs_dir_, test=test["load_mwmed"])
    

# COLOCAR RESTANTE DOS MODELOS
@timer_decorator
def run_models():
    print("RODANDO MODELOS")
    print("\t#### Prophet ####")
    fc_p = fm.prophet_fit_forecast()
    print("\t#### Holt-Winters ####")
    fc_hw = fm.hw_fit_forecast()
    print("\t#### MSTL ####")
    fc_ml = fm.mstl_fit_forecast()
    print("\t#### AutoARIMA ####")
    fc_aa = fm.auto_arima_fit_forecast()
