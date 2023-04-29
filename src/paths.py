import os


# Define the root directory of your project
ROOT_DIR = ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#ROOT_DIR = "C:\\Users\\user\\Projetos\\load_forecasting\\src"
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "01_external")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "02_interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "03_processed")
FORECASTS_DATA_DIR = os.path.join(DATA_DIR, "04_forecasts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FORECASTS_FIG_DIR = os.path.join(ROOT_DIR, "reports", "figures", "forecasts")

PATHS = {
    'root': ROOT_DIR,
    'data': DATA_DIR,
    'external_data': EXTERNAL_DATA_DIR,
    'interim_data': INTERIM_DATA_DIR,
    'processed_data': PROCESSED_DATA_DIR,
    'forecasts_data': FORECASTS_DATA_DIR,
    'models': MODELS_DIR,
    'forecasts_figs': FORECASTS_FIG_DIR
    #'logs': LOGS_DIR,
    #'config': CONFIG_FILE,
    #'readme': README_FILE,
    #'pandas': PANDAS_PATH,
    #'numpy': NUMPY_PATH
}