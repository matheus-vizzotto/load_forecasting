import logging

logging.basicConfig(level=logging.WARNING,
                    filename='lab/matheus/notebooks/logs/data_info.log',
                    filemode='w',
                    format = '%(asctime)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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
