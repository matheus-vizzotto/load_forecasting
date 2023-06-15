import hydra
from omegaconf import DictConfig

@hydra.main(config_path='utils', config_name='config', version_base=None)
def get_dates(config: DictConfig):
    print(config.DATA_PARAMS.START_DATE)

if __name__ == '__main__':
    get_dates()