import yaml

from config.config import CONFIG_DIR

def load_config(config_path=CONFIG_DIR / 'config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config