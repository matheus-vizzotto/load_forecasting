import scraper
import forecaster
import pandas as pd
import metrics
from paths import PATHS
import matplotlib.pyplot as plt
import os
from datetime import datetime

FCS_PATH = PATHS["forecasts_data"]
FORECASTS_DATA_DIR = PATHS['forecasts_data'] 

# DOWNLOAD DATA
scraper.run_download()
# RUN MODELS
forecasts = forecaster.run_models()
# PLOT FORECASTS
plt.figure(figsize=(15,5))
for i in range(len(forecasts)):
    plt.plot(forecasts[i][1]["datetime"], forecasts[i][1]["yhat"], label=forecasts[i][0])
plt.title("Projeção para os próximos dias")
plt.legend()
plt.show()
# GENERATE forecasts.xlsx FILE
df_forecasts = pd.DataFrame()
for model in forecasts:
    df_sub = model[1]
    df_sub["model"] = model[0]
    df_sub["ingestion_date"] = datetime.now().strftime("%Y%m%d")
    df_forecasts = pd.concat([df_forecasts, df_sub])
file_path = os.path.join(FCS_PATH, "forecasts.xlsx")
df_forecasts.to_excel(file_path, index=False)
# GET THE MODEL WITH THE LOWEST ERROR METRICS
all_metrics = forecaster.run_evaluation()
best_model = metrics.get_best_model(all_metrics)
print(f"\nSeu melhor modelo é o {best_model.upper()}.")

print("\nPrograma executado com sucesso! Volte sempre!")
 