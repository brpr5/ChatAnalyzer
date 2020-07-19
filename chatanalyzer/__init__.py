# import warning
import logging
import logging.config
from logging import NullHandler
from chatanalyzer.utils import constants
# # https://michaelheap.com/using-ini-config-with-python-logger/

logging.config.fileConfig('logging_config.ini', disable_existing_loggers=False)

# # create logger
log = logging.getLogger(constants.logger_name)
log.addHandler(logging.NullHandler())
