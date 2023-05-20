import logger as l
import os

print(os.getcwd())
ano = 2020

l.log_data_info(estacao="estacao teste", ano=ano, nans=1000)