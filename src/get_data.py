import utils.data_wrangling as dw
import os

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

def run_download():
    print("DADOS DE CARGA ELÉTRICA")
    load_data = dw.ons_data(freq='h', ano_inicio=2012, ano_fim=2023, idreg="S")
    load_data.update(printer=True, write=True)

    print("\nDADOS METEOROLÓGICOS")
    temp_data = dw.inmet_data(2012,2023)
    temp_data.download()
    temp_data.build_database()
    temp_data.aggregate_by_hour()

# run_download()