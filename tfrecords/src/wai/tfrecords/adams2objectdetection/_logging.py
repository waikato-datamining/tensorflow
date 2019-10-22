"""
Sets up the logger for this package.
"""
import logging
logging.basicConfig()

# The name of the logger for this package
LOGGING_NAME: str = __package__

# Create the logger
logger = logging.getLogger(LOGGING_NAME)
logger.setLevel(logging.INFO)
