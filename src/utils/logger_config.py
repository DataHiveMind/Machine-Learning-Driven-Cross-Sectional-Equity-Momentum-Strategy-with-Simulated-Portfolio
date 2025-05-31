import logging
import logging.config
import yaml
from pathlib import Path

def setup_logger(config_path: str = "config/logging_config.yaml", default_level=logging.INFO):
    """
    Set up logging configuration from a YAML file.
    If the config file is missing or invalid, falls back to basic config.
    """
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            print(f"Failed to load logging config: {e}. Using basicConfig.")
            logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)

def get_logger(name: str = None):
    """
    Get a logger by name. If name is None, returns the root logger.
    """
    return logging.getLogger(name)

# Example usage (uncomment for testing):
# setup_logger()
# logger = get_logger(__name__)
# logger.info("Logger initialized.")