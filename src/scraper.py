import utils.data_wrangling as dw
import os
#from utils.data_integrity import measure_time
from utils.logger import timer_decorator

# IDENTIFICADORES
ANO_INICIO = 2012
ANO_FIM = 2023

@timer_decorator
def run_download():
    print("INICIANDO DOWNLOAD DE DADOS")
    print("\t-CARGA ELÉTRICA")
    load_data = dw.ons_data(freq='h', ano_inicio=ANO_INICIO, ano_fim=ANO_FIM, idreg="S")
    load_data.update(printer=True, write=True)

    # print("\t-METEOROLOGIA")
    # temp_data = dw.inmet_data(ano_inicio=ANO_INICIO, ano_fim=ANO_FIM)
    # temp_data.download()
    # temp_data.build_database()
    # temp_data.read_parquet()
    # temp_data.aggregate_by_hour()

# run_download()