import logging

def setup_logger(log_file='app.log'):
    logging.basicConfig(
        level=logging.DEBUG,  # Log all levels of messages
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
