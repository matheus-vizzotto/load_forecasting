import get_data
import forecaster
from paths import PATHS
import os


FCS_PATH = PATHS["forecasts_data"]


#get_data.run_download()
forecaster.run_models(FCS_PATH)

print("\nPrograma executado com sucesso! Volte sempre!")
