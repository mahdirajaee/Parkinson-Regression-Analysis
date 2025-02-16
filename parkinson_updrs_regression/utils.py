import logging

def setup_logger():
    """
    Configures logging for the application.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_message(message):
    """
    Logs a message to the console.
    
    Args:
        message (str): Message to be logged.
    """
    logging.info(message)
