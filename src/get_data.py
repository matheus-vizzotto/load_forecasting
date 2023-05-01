import utils.data_wrangling as dw
import os
#from utils.data_integrity import measure_time
from utils.logger import timer_decorator

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

#@measure_time
@timer_decorator
def run_download():
    print("Iniciando download de dados.")
    print("\tDADOS DE CARGA ELÉTRICA")
    load_data = dw.ons_data(freq='h', ano_inicio=2022, ano_fim=2023, idreg="S")
    load_data.update(printer=True, write=True)

    print("\n\tDADOS METEOROLÓGICOS")
    temp_data = dw.inmet_data(2022,2023)
    temp_data.download()
    temp_data.build_database()
    temp_data.read_parquet()
    temp_data.aggregate_by_hour()

# run_download()