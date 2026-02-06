from loguru import logger
import sys

def setUpLogger(level= str):

    logger.remove()

    logger.add(sys.stdout,
            # colorize=True,
            level=   level,
            format=     "<level>{level: <10}</level> " \
                        "{name: <8} " \
                        "{function: <8} " \
                        "{line: <3} "
                        "<level>{message}</level>")