import scraper
import forecaster
from paths import PATHS
import matplotlib.pyplot as plt
import os

#FCS_PATH = PATHS["forecasts_data"]

#scraper.run_download()
forecasts = forecaster.run_models()
plt.figure(figsize=(15,5))
for i in range(len(forecasts)):
    plt.plot(forecasts[i][1], label=forecasts[i][0])
plt.legend()
plt.show()
#forecaster.run_cross_validation()
print("\nPrograma executado com sucesso! Volte sempre!")
 