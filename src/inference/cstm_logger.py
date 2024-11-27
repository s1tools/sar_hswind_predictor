import logging
from logging.handlers import TimedRotatingFileHandler
import socket


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.hostname = socket.gethostname()
        return True


def create_logger(logfilename, verbose=1):
    """
    Aim of method : create a SOFT log object to write information about the get_coloc_mfwam.
    Args:
        :param logfile (string)  :filepath of the logfile to create

        :param verbose (boolean) :Set the level of console verbose
    Return
        logger object to write information
    """
    # __CREATE LOGGER
    logger = logging.getLogger('asyncio')  ## ASYNCIO to avoid too many unnessary debug messages
    # logger = logging.getLogger()

    f = ContextFilter()
    logger.addFilter(f)

    # __MESSAGE FORMAT
    formatter = logging.Formatter(fmt='{asctime} | @{hostname} | {levelname} | {message}', style="{")

    file_handler = TimedRotatingFileHandler(logfilename, when='d', interval=1, backupCount=5)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # __HANDLING LOGFILE
    if verbose > 1:
        logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
        file_handler.setLevel(logging.ERROR)
        stream_handler.setLevel(logging.ERROR)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    # logging.captureWarnings(True)
    # warnings_logger = logging.getLogger("py.warnings")
    # warnings_logger.addHandler(file_handler)
    return logger
