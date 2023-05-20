import streamlit as st

logo_link = "https://www.pngmart.com/files/15/Energy-Symbol-Transparent-PNG.png"

with st.sidebar:
    st.image(logo_link)
    st.title("Projeção de demanda elétrica")
    choice = st.radio("Navegação", ["Download - Carga elétrica", "Análise exploratória", "Modelos", "Performance", "Projeções"])
    st.info("Aplicativo de projeção de demanda elétrica com modelos de aprendizado de máquina e modelos econométricos.")

st.write("Hello world")