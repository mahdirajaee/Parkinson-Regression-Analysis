import logging

def setup_logger():
    """Sets up the logger for tracking execution."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def log_message(message):
    """Logs an information message."""
    logging.info(message)
