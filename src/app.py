import streamlit as st
from paths import PATHS
from forecaster import load_data
import plotly_express as px

PROCESSED_DATA_DIR = PATHS['processed_data']
LOGO_LINK = "https://www.pngmart.com/files/15/Energy-Symbol-Transparent-PNG.png"

with st.sidebar:
    st.image(LOGO_LINK)
    st.title("Projeção de demanda elétrica")
    choice = st.radio("Navegação", ["Dados", "Análise exploratória", "Modelos", "Performance", "Projeções"])
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
    st.title("Análise exploratória")
    st.subheader(f"Período: {ts.data.index.min()} a {ts.data.index.max()}")
    level_plot = px.line(ts.data, y=["load_mwmed"])
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

if choice=="Modelos":
    pass

if choice=="Performance":
    pass

if choice=="Projeções":
    pass