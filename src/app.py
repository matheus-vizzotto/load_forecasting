import streamlit as st
from paths import PATHS
from forecaster import load_data
from utils.plots import get_n_lags_plot, plot_metrics
import plotly.io as pio
import os
from utils.ts_wrangling import seasonal_decompose, get_acf_pacf
import pandas as pd
import plotly_express as px

PROCESSED_DATA_DIR = PATHS['processed_data']
LOGO_LINK = "https://www.pngmart.com/files/15/Energy-Symbol-Transparent-PNG.png"
FORECASTS_DIR = PATHS["forecasts_data"]

with st.sidebar:
    st.image(LOGO_LINK)
    st.title("Projeção de demanda elétrica")
    choice = st.radio("Navegação", ["Dados", "Análise exploratória", "Performance", "Projeções"])
    st.info("Aplicativo de projeção de demanda elétrica com modelos de aprendizado de máquina e modelos econométricos.")

ts = load_data()
if choice=="Dados":
    st.title("Dados")
    st.write("Escolha uma forma de importação de dados:")
    web_choice = st.button(":spider_web: Fazer download de dados")
    local_choice = st.button(":open_file_folder: Ler arquivo local")
    if local_choice:
        ts_class = ts
        st.dataframe(ts.data)
    elif web_choice:
        pass

if choice=="Análise exploratória":
    st.title("Análise exploratória :flashlight:")
    st.subheader(f"Período: {ts.data.index.min()} a {ts.data.index.max()}")
    level_plot = px.line(ts.data, y=["load_mwmed"], template="plotly_dark")
    level_plot.update_layout(title='Série em nível')
    st.plotly_chart(level_plot)
    seasonal_data = ts.seasonal_components
    for col in seasonal_data.columns:
        if (col=="data") or (col=="load_mwmed"):
            continue
        else:
            plot = px.scatter(seasonal_data, x=col, y="load_mwmed", hover_data=[seasonal_data["data"]])
            plot.update_layout(title=col)
            # median = seasonal_data.groupby(col)["load_mwmed"].median()
            # plot.add_trace(px.line(median))
            st.plotly_chart(plot)
    hist_plot = px.histogram(ts.data.load_mwmed)
    hist_plot.update_layout(title="Histograma de valores")
    st.plotly_chart(hist_plot)
    st.subheader("Decomposição da série - MSTL")
    last_obs = 24*30
    decomposed_df = seasonal_decompose(data=ts.data.reset_index(), y_col="load_mwmed", date_col="date").tail(last_obs)
    for col in decomposed_df.columns:
        if col=="data":
            continue
        else:
            plot = px.line(x=ts.data.index[-last_obs:], y=decomposed_df.loc[:,col], hover_data=[seasonal_data["dia_semana"].tail(last_obs)])
            plot.update_layout(title=col)
            st.plotly_chart(plot)
    st.subheader("Análise de autocorrelação")
    df_autocorr_fun = get_acf_pacf(ts.data, y_col="load_mwmed")
    acf_plot = px.bar(df_autocorr_fun, x='Lag', y='ACF', title='Autocorrelation Function (ACF)')
    st.plotly_chart(acf_plot)
    pacf_plot = px.bar(df_autocorr_fun, x='Lag', y='PACF', title='Autocorrelation Function (ACF)')
    st.plotly_chart(pacf_plot)
    y_lags = st.selectbox('Selecione o número de lags', options=[x for x in range(1,24*14)])
    lag_corr_plot = get_n_lags_plot(ts.data, y_col="load_mwmed", n_lags=y_lags, hue=ts.seasonal_components["dia_semana"])
    st.plotly_chart(lag_corr_plot)

# if choice=="Modelos":
#     pass

if choice=="Performance":
    st.title("Performance dos modelos :medal:")
    st.subheader("Projeção fora de amostra")
    #df_oos_fcs = pd.read_excel(os.path.join(FORECASTS_DIR, "forecasts.xlsx"))
    df_oos_fcs = pd.read_parquet(os.path.join(FORECASTS_DIR, "fc_vs_test.parquet"))
    oos_dates = df_oos_fcs["datetime"]
    h_min, h_max, h = oos_dates.min(), oos_dates.max(), oos_dates.nunique()
    st.write(f"De {h_min} a {h_max} (T = {h})")
    df_oos_test = df_oos_fcs[["datetime", "y"]].drop_duplicates()
    df_oos_test["model"] = "Observado"
    df_oos_test.rename(columns={"y": "yhat"}, inplace=True)
    df_oos_fcs = pd.concat([df_oos_fcs, df_oos_test])
    st.dataframe(df_oos_fcs)
    oos_fcs_plot = px.line(df_oos_fcs, x="datetime", y="yhat", color="model")
    oos_fcs_plot.update_layout(hovermode="x")
    st.plotly_chart(oos_fcs_plot)
    df_individual_forecasts = pd.read_parquet(os.path.join(FORECASTS_DIR, "fc_vs_test.parquet"))
    models_names = df_individual_forecasts["model"].unique()
    models_checkbox = st.selectbox('Selecione o modelo de previsão', options=models_names)
    df_individual_forecasts_checkbox = df_individual_forecasts[df_individual_forecasts["model"]==models_checkbox]
    error_hist = px.histogram(df_individual_forecasts_checkbox, x="error", color="model", opacity=0.5)
    st.plotly_chart(error_hist)
    df_cum_met = pd.read_parquet(os.path.join(FORECASTS_DIR, "cummulative_metrics.parquet"))
    metrics_opts = [x for x in df_cum_met.columns if x not in ("model", "i")]
    metric = st.selectbox('Selecione a métrica de avaliação', options=metrics_opts)
    metrics_plot = plot_metrics(data_=df_cum_met, x_="i", y_= metric, hue_="model")
    metrics_plot.update_layout(hovermode="x")
    st.plotly_chart(metrics_plot)

if choice=="Projeções":
    st.title("Projeções :crystal_ball:")
    df_final_forecasts = pd.read_excel(os.path.join(FORECASTS_DIR, "forecasts.xlsx"))
    st.dataframe(df_final_forecasts)
    final_forecasts_plot = px.line(df_final_forecasts, x="datetime", y="yhat", color="model")
    st.plotly_chart(final_forecasts_plot)