import logging
import os
import psutil

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the log level you desire
    if not logger.handlers:  # Check if handlers have already been added
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False  # Prevent the logger from propagating messages to ancestor loggers
    return logger



def log_memory_usage(logger):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")

# Ensure root logger is also configured
setup_logger('root')
