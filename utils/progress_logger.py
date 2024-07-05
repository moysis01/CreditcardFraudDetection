import logging

class ProgressLogger:
    def __init__(self, logger, total_steps):
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0

    def log_step(self, message):
        self.current_step += 1
        progress_percentage = (self.current_step / self.total_steps) * 100
        self.logger.info(f"{message} ({progress_percentage:.2f}%)")
