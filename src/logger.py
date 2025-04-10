import logging

def get_logger(name):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
        level=logging.INFO
    )
    return logging.getLogger(name)
