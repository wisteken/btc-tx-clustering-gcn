import sys
import logging
from pytz import timezone
from datetime import datetime


def logger(name: str = None, logdir: str = '../logs'):
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level=logging.INFO)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    filename = f"{logdir}/{name+'_' if name else ''}{datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d_%H:%M:%S')}.log"
    logfile_handler = logging.FileHandler(filename=filename)
    logfile_handler.setLevel(level=logging.DEBUG)
    logfile_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(stdout_handler)
    logger.addHandler(logfile_handler)

    return logger
