import scraper
import forecaster
import pandas as pd
from paths import PATHS
import matplotlib.pyplot as plt
import os
from datetime import datetime

FCS_PATH = PATHS["forecasts_data"]

scraper.run_download()
forecasts = forecaster.run_models()

plt.figure(figsize=(15,5))
for i in range(len(forecasts)):
    plt.plot(forecasts[i][1], label=forecasts[i][0])
plt.legend()
plt.show()

df_forecasts = pd.DataFrame()
for model in forecasts:
    df_sub = model[1].reset_index()
    df_sub.columns = ["ds", "yhat"]
    df_sub["model"] = model[0]
    df_sub["ingestion_date"] = datetime.now().strftime("%Y%m%d")
    df_forecasts = pd.concat([df_forecasts, df_sub])
file_path = os.join(FCS_PATH, "forecasts.parquet")
df_forecasts.to_parquet(file_path)


print("\nPrograma executado com sucesso! Volte sempre!")
 