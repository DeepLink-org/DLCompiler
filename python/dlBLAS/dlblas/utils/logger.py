import logging


class ColorFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[1;32m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # fformat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    fformat = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + fformat + reset,
        logging.INFO: green + fformat + reset,
        logging.WARNING: yellow + fformat + reset,
        logging.ERROR: red + fformat + reset,
        logging.CRITICAL: bold_red + fformat + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.INFO)  # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger