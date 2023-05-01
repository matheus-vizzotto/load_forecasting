import logging
import time
from paths import PATHS
import os
import datetime as dt

LOGS_PATH = PATHS['logs']
DATA_INFO_PATH = os.path.join(LOGS_PATH, "data_info.log")
MODELS_INFO_PATH = os.path.join(LOGS_PATH, "models_info.log")
TIMING_INFO_PATH = os.path.join(LOGS_PATH, "timing_info.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# logging.basicConfig(level=logging.WARNING,
#                     #filename='lab/matheus/notebooks/logs/data_info.log',
#                     #filename='C:/Users/user/Projetos/load_forecasting/tests/matheus/notebooks/logs/data_info.log', # TODO: AJUSTAR PARA CAMINHO RELATIVO
#                     filename=DATA_INFO_PATH, 
#                     filemode='w',
#                     format = '%(asctime)s - %(message)s')

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

data_info_logger = setup_logger('data_logs', DATA_INFO_PATH, level=logging.INFO)

def log_data_info(estacao: str, ano: int, nans: int):
    """_summary_

    Args:
        message (_type_): _description_
        estacao (str): _description_
        ano (int): _description_
        nans (int): _description_
        cells (int): _description_
        date_min (dt.datetime): _description_
        date_max (dt.datetime): _description_
    """
    data_info_logger.info(f"[INMET] [RESUMO] Estação {estacao}, ano {ano}: VALORES VAZIOS: {nans:,}")# | DATA MÍNIMA: {date_min} | DATA MÁXIMA: {date_max}")

def log_dates(estacao: str, ano: int, missing_dates: list, datas_estranhas: list):
    """_summary_

    Args:
        estacao (str): _description_
        ano (int): _description_
        missing_dates (list): _description_
        datas_estranhas (list): _description_
    """
    data_info_logger.info(f"[INMET] [RESUMO] Estação {estacao}, ano {ano}: DATAS FALTANTES: {missing_dates} | DATAS ESTRANHAS: {datas_estranhas}")

def general_data_info_log(message):
    data_info_logger.info(f"{message}")


timing_logs = setup_logger('timing_logs', TIMING_INFO_PATH, level=logging.INFO)

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        delta_time = dt.timedelta(seconds=total_time)
        ref_date = dt.datetime(2023, 4, 29, 0, 0, 0)  # fixed reference date
        time_str = (ref_date + delta_time).strftime("%H:%M:%S")
        #logging.warning(f'{func.__name__} executed in {end_time - start_time} seconds')
        timing_logs.info(f'{func.__name__} executado em {time_str} (HH:MM:SS)')
        return result
    return wrapper