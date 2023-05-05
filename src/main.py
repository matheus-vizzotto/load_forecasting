import scraper
import forecaster
from paths import PATHS
import os

#FCS_PATH = PATHS["forecasts_data"]

scraper.run_download()
forecaster.run_models()

print("\nPrograma executado com sucesso! Volte sempre!")
 