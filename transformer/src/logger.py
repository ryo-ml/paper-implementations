import sys
import logging

def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger