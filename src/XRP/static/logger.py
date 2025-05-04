import logging
import os

def setup_logger(name=__name__, log_file='app.log', level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Crear carpeta logs si no existe
    if not os.path.exists('logs'):
        os.makedirs('logs')

    handler = logging.FileHandler(os.path.join('logs', log_file))
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger
