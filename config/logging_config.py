import os
import yaml
from loguru import logger

# yaml file for config all new sinks require update this file
CONFIG_PATH = "/home/mdp23/MDP_RPI/config/logging.yaml"
loggers = {}


# set up loggers
def setup_loggers():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    all_loggers = config.get("loggers", {})

    for module_name, cfg in all_loggers.items():
        log_path = cfg.get("path")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

        rotation = cfg.get("rotation", "1 week")
        retention = cfg.get("retention", None)
        level = cfg.get("level", "INFO")
        fmt = cfg.get("format", "{time} | {level} | {message}")
        backtrace = cfg.get("backtrace", True)
        diagnose = cfg.get("diagnose", True)
        compression = cfg.get("compression", None)

        logger.add(
            log_path,
            rotation=rotation,
            retention=retention,
            level=level,
            format=fmt,
            backtrace=backtrace,
            diagnose=diagnose,
            compression=compression,
        )

        bound_logger = logger.bind(module=module_name)
        loggers[module_name] = bound_logger


setup_loggers()
