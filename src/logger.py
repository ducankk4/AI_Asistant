import logging

def setup_logger():
    logging.basicConfig(
        level= logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename= 'app.log',
        filemode= 'w'
    )
    logger = logging.getLogger()
    return logger

logger = setup_logger()
