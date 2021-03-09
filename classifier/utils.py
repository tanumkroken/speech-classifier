#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#

import logging
import colorlog

DELIMITER = 52*'-'
SECTION = 52*'='
INDENT = 10*' '
INDENT_SMALL = 2*' '
RIGHT_PAD = ':<45'


def init_logger(app_name, multiple: bool, loglevel='INFO') -> logging.Logger:
    logger = logging.getLogger(app_name)
    # Set the logging level
    if loglevel == 'ERROR':
        logger.setLevel(logging.ERROR)
        log_format = (
            '%(asctime)s: '
            '%(name)s - '
            '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fhlog_format = (
            '%(asctime)s: '
            '%(name)s - '
            #  '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fh_error_log_format = fhlog_format
        level = logging.ERROR
    elif loglevel == 'WARNING':
        logger.setLevel(logging.WARNING)
        log_format = (
            '%(asctime)s: '
            '%(name)s - '
            # '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fhlog_format = (
            '%(asctime)s: '
            '%(name)s - '
            '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fh_error_log_format = fhlog_format
    elif loglevel == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        log_format = (
            '%(asctime)s: '
            '%(name)s - '
            '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fhlog_format = (
            '%(asctime)s: '
            '%(name)s - '
            '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fh_error_log_format = fhlog_format
    else:
        logger.setLevel(logging.INFO)
        log_format = (
            '%(asctime)s: '
            # '%(name)s - '
            #'%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
        fhlog_format = (
            # '%(asctime)s: '
            # '%(name)s - '
            # '%(funcName)30s - '
            # '%(levelname)8s: '
            '%(message)s'
        )
        fh_error_log_format = (
            # '%(asctime)s: '
            # '%(name)s - '
            # '%(funcName)30s - '
            '%(levelname)8s: '
            '%(message)s'
        )
    # Console color settings
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)


    # Output separate logs
    if multiple:

        # Output warning log
        fh = logging.FileHandler(app_name+'.warning.log')
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(fh_error_log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Output error log
        fh = logging.FileHandler(app_name+'.error.log')
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(fh_error_log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Info log
        fh = logging.FileHandler(app_name + '.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(fhlog_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        # Output one logfile
        fh = logging.FileHandler(app_name + '.log')
        fh.setLevel(logger.level)
        formatter = logging.Formatter(fhlog_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



class MsgLine:
    def __init__(self, msg: str, parameter, format: str = ''):
        formatted = '{'+RIGHT_PAD+'}: ' + '{' + format + '}'
        self.message = formatted.format(msg,parameter)

    def msg(self):
        return self.message

class MsgSection:
    def __init__(self, msg: str, logger):
        self.message = msg
        self.logger = logger

    def log(self):
        self.logger.info(' ') # A blank line above each section
        self.logger.info(self.message)
        self.logger.info(DELIMITER)

class MsgHeader:
    def __init__(self, msg: str, logger):
        self.message = msg
        self.logger = logger

    def log(self):
        self.logger.info(SECTION)
        self.logger.info(self.message)
        self.logger.info(SECTION)