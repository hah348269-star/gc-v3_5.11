import logging
import os
from datetime import datetime


class LogUtils:
    @staticmethod
    def get_logger(name='carbon_system'):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d")}.log'),
                                 encoding='utf-8')
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger