import requests
from io import BytesIO
import pandas as pd
from zipfile import ZipFile
import re
from dtype_diet import report_on_dataframe, optimize_dtypes
import os
from dataprep.clean import clean_headers
import warnings
warnings.filterwarnings('ignore')


columns_dict = {
    'DATA (YYYY-MM-DD)': 'data',
    'Data': 'data',
    'HORA (UTC)': 'hora',
    'Hora UTC': 'hora',
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_horario_mm',
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_horario_mm',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'pressao_atmosferica_max_na_hora_ant_aut_m_b',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'pressao_atmosferica_min_na_hora_ant_aut_m_b',
    'RADIACAO GLOBAL (KJ/m²)': 'radiacao_global_kj_m',
    'RADIACAO GLOBAL (Kj/m²)': 'radiacao_global_kj_m',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temperatura_do_ar_bulbo_seco_horaria_c',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'temperatura_do_ponto_de_orvalho_c',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'temperatura_maxima_na_hora_ant_aut_c',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'temperatura_minima_na_hora_ant_aut_c',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'temperatura_orvalho_max_na_hora_ant_aut_c',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'temperatura_orvalho_min_na_hora_ant_aut_c',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'umidade_rel_max_na_hora_ant_aut_%',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'umidade_rel_min_na_hora_ant_aut_%',
    'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade_relativa_do_ar_horaria_%',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'vento_direcao_horaria_gr_gr',
    'VENTO, RAJADA MAXIMA (m/s)': 'vento_rajada_maxima_m_s',
    'VENTO, VELOCIDADE HORARIA (m/s)': 'vento_velocidade_horaria_m_s',
    'ESTACAO': 'estacao',
    'UF': 'uf',
    'REGIAO': 'regiao',
}

col_types = {
    'precipitacao_total_horario_mm': float,
    'pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b': float,
    'pressao_atmosferica_max_na_hora_ant_aut_m_b': float,
    'pressao_atmosferica_min_na_hora_ant_aut_m_b': float,
    'radiacao_global_kj_m': float,
    'temperatura_do_ar_bulbo_seco_horaria_c': float,
    'temperatura_do_ponto_de_orvalho_c': float,
    'temperatura_maxima_na_hora_ant_aut_c': float,
    'temperatura_minima_na_hora_ant_aut_c': float,
    'temperatura_orvalho_max_na_hora_ant_aut_c': float,
    'temperatura_orvalho_min_na_hora_ant_aut_c':float,
    'umidade_rel_max_na_hora_ant_aut_%': float,
    'umidade_rel_min_na_hora_ant_aut_%': float,
    'umidade_relativa_do_ar_horaria_%': float,
    'vento_direcao_horaria_gr_gr': float,
    'vento_rajada_maxima_m_s': float,
    'vento_velocidade_horaria_m_s': float,
    'estacao': 'category',
    'uf': 'category',
    'regiao': 'category'
    }

def ajusta_hora(x):
    if ":" in x:
        y = x
    elif "UTC" in x:
        y = x[:2] + ":" + x[2:4]
    else:
        y = None
    return y

def converte_data(x):
    try:
        y = pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S')
    except:
        y = pd.to_datetime(x, format = '%Y/%m/%d %H:%M:%S')
    return y

df = pd.DataFrame()
for ano in range(2010, 2024):
    print(ano)
    path = f'https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip'
    r = requests.get(path, verify = False)
    files = ZipFile(BytesIO(r.content))
    arquivos = [file for file in files.namelist() if file.lower().endswith(".csv")]

    #df01 = pd.DataFrame()
    for arquivo in arquivos:
        info = pd.read_csv(files.open(arquivo), sep = ";", encoding = "latin-1", nrows=7, header = None)
        info2 = {line[1][0]: line[1][1] for line in info.iterrows()}
        #df02 = pd.read_csv(files.open(arquivo),  sep = ";", encoding = "latin-1", skiprows = 8, nrows=1)
        df02 = pd.read_csv(files.open(arquivo),  sep = ";", encoding = "latin-1", skiprows = 8)
        df02.rename(columns=columns_dict, inplace=True)
        #df02["estacao"] = info2['ESTACAO:']
        df02["estacao"] = info.iloc[2,1]
        df02["uf"] = info.iloc[1,1]
        df02["regiao"] = info.iloc[0,1]
        #print(info.iloc[2,1], info.iloc[1,1], info.iloc[0,1])
        for col in df02.columns:
            df02.loc[:,col] = df02.loc[:,col].replace(",",".", regex=True)
        df02 = df02.astype(col_types)
        #df02.loc[:, "hora"] = df02.loc[:, "hora"].apply(ajusta_hora)
        #df02["data_hora"] = df02["data"] + " " + df02["hora"]
        #df02.loc[:, "data_hora"] = df02.loc[:, "data_hora"].apply(converte_data)
        if "Unnamed: 19" in df02.columns:
            df02.drop(["Unnamed: 19"], axis=1, inplace=True) 
        #df01 = pd.concat([df01, df02])
        df = pd.concat([df, df02])
        #print(df["data_hora"].iloc[-1], df["estacao"].iloc[-1])
        print("\t",df["estacao"].iloc[-1])
        #for col in df02.columns.to_list():
        #    nomes_colunas.append(col)
    #df = pd.concat([df, df01])


df.loc[:, "hora"] = df.loc[:, "hora"].apply(ajusta_hora)
df["data_hora"] = df["data"] + " " + df["hora"]
df.loc[:, "data_hora"] = df.loc[:, "data_hora"].apply(converte_data)

df.to_parquet("inmet_data.parquet")