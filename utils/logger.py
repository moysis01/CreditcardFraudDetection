import logging
import os
import psutil

class ProgressFilter(logging.Filter):
    def filter(self, record):
        return 'Results' not in record.getMessage()

class ResultsFilter(logging.Filter):
    def filter(self, record):
        return 'Results' in record.getMessage()

def setup_logger(name, log_file='results.log', console_level=logging.INFO, file_level=logging.INFO):
    """
    Sets up a logger with the specified name.

    Args:
    - name (str): The name of the logger.
    - log_file (str, optional): Path to a log file. Defaults to 'results.log'.
    - console_level (int, optional): Logging level for the console handler. Defaults to logging.INFO.
    - file_level (int, optional): Logging level for the file handler. Defaults to logging.INFO.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # overall logger level to the lowest needed

    if not logger.handlers:  
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.addFilter(ProgressFilter())
        ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        if log_file:
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(file_level)
            fh.addFilter(ResultsFilter())
            fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fh_formatter)
            logger.addHandler(fh)

    logger.propagate = False  # Preventing the logger from propagating messages to ancestor loggers
    return logger


def log_memory_usage(logger):
    """
    Logs the current memory usage of the process.

    Args:
    - logger (logging.Logger): Logger instance to use for logging.
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")

