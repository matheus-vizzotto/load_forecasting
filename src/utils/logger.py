import logging
import time
from paths import PATHS
import os

LOGS_PATH = PATHS['logs']
DATA_INFO_PATH = os.path.join(LOGS_PATH, "data_info.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

logging.basicConfig(level=logging.WARNING,
                    #filename='lab/matheus/notebooks/logs/data_info.log',
                    #filename='C:/Users/user/Projetos/load_forecasting/tests/matheus/notebooks/logs/data_info.log', # TODO: AJUSTAR PARA CAMINHO RELATIVO
                    filename=DATA_INFO_PATH, 
                    filemode='w',
                    format = '%(asctime)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

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
    logger.warning(f"[RESUMO] Estação {estacao}, ano {ano}: VALORES VAZIOS: {nans:,}")# | DATA MÍNIMA: {date_min} | DATA MÁXIMA: {date_max}")

def log_dates(estacao: str, ano: int, missing_dates: list, datas_estranhas: list):
    """_summary_

    Args:
        estacao (str): _description_
        ano (int): _description_
        missing_dates (list): _description_
        datas_estranhas (list): _description_
    """
    logger.warning(f"[RESUMO] Estação {estacao}, ano {ano}: DATAS FALTANTES: {missing_dates} | DATAS ESTRANHAS: {datas_estranhas}")


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.warning(f'{func.__name__} executed in {end_time - start_time} seconds')
        return result
    return wrapper