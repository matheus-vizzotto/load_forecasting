import scraper
import forecaster
import pandas as pd
from paths import PATHS
import matplotlib.pyplot as plt
import os
from datetime import datetime
from metrics import get_cumulative_metrics

FCS_PATH = PATHS["forecasts_data"]
FORECASTS_DATA_DIR = PATHS['forecasts_data'] 

#scraper.run_download()
forecasts = forecaster.run_models()

plt.figure(figsize=(15,5))
for i in range(len(forecasts)):
    plt.plot(forecasts[i][1]["datetime"], forecasts[i][1]["yhat"], label=forecasts[i][0])
plt.title("Projeção para os próximos dias")
plt.legend()
plt.show()

df_forecasts = pd.DataFrame()
for model in forecasts:
    df_sub = model[1]
    df_sub["model"] = model[0]
    df_sub["ingestion_date"] = datetime.now().strftime("%Y%m%d")
    df_forecasts = pd.concat([df_forecasts, df_sub])
file_path = os.path.join(FCS_PATH, "forecasts.parquet")
df_forecasts.to_parquet(file_path)

# TODO: PASSAR PARA MÓDULO "FORECASTER"
# EVALUATION 
oos_forecasts = [os.path.join(FORECASTS_DATA_DIR, file) for file in os.listdir(FORECASTS_DATA_DIR) if file.startswith('oos')]
df_oos_forecasts = pd.DataFrame()
for oos_forecast in oos_forecasts:
    df_oos_forecast = pd.read_parquet(oos_forecast)
    df_oos_forecast["model"] = oos_forecast.split("\\")[-1].split("_")[1]
    df_oos_forecasts = pd.concat([df_oos_forecasts, df_oos_forecast])
df_fc_test = pd.merge(df_oos_forecasts, ts.full_series, left_on="datetime", right_index=True, how="left")
df_fc_test.rename(columns={"load_mwmed": "y"}, inplace=True)
cum_metrics = []
for model in df_fc_test.model.unique():
    df_model_sub = df_fc_test[df_fc_test["model"]==model]
    metrics = get_cumulative_metrics(df_model_sub["yhat"], df_model_sub["y"])
    cum_metrics.append(metrics)
df_final = pd.DataFrame(cum_metrics)

print("\nPrograma executado com sucesso! Volte sempre!")
 