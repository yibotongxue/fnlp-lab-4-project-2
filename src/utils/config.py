from typing import Any
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict[str, Any]: Loaded configuration.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
