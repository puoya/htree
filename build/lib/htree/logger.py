# htree/logger.py
import logging
import os
from datetime import datetime
from . import conf
# import conf

# Global logging state
_LOGGING_ENABLED = False
_CURRENT_TIME = None  

def set_logger(enable: bool, log_dir: str = conf.LOG_DIRECTORY, log_level: int = logging.INFO):
    """
    Enable or disable logging for the `htree` package.

    This function sets up logging for the `htree` package based on the provided parameters.
    When logging is enabled, it creates a log file in the specified directory and logs
    messages at the specified logging level. When logging is disabled, it stops logging
    and clears existing log handlers.

    Parameters:
        enable (bool): Whether to enable logging.
        log_dir (str): Directory to store log files.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Example usage:
        import logger
        logger.set_logger(True)
    """
    global _LOGGING_ENABLED
    global _CURRENT_TIME
    _LOGGING_ENABLED = enable  

    if enable and _CURRENT_TIME is None:
        _CURRENT_TIME = datetime.now()  # Set the current time once when logging is enabled

    logger = logging.getLogger("htree")

    if enable:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"htree_{_CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}.log")

        # Remove existing handlers to prevent duplicate logs
        logger.handlers.clear()

        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger.setLevel(log_level)
        logger.info("Logging enabled for htree package.")
    else:
        # Log the fact that logging is being disabled
        logger.info("Disabling logging for htree package.")
        
        # Remove all handlers to prevent logs from being written
        logger.handlers.clear()
        logger.setLevel(logging.CRITICAL + 1)  # Effectively disables new logs

def get_logger():
    """
    Return the global logger instance for the `htree` package.

    This function retrieves the logger instance for the `htree` package, which can be used
    to log messages at various levels (e.g., info, warning, error).

    Example usage:
        import logger
        logger_instance = logger.get_logger()
        logger_instance.info("This is an informational message.")
    """
    return logging.getLogger("htree")

def logging_enabled():
    """
    Check if logging is currently enabled.

    This function returns the current logging status, indicating whether logging is enabled
    or disabled for the `htree` package.

    Returns:
        bool: True if logging is enabled, False otherwise.

    Example usage:
        import logger
        if logger.logging_enabled():
            print("Logging is enabled.")
        else:
            print("Logging is disabled.")
    """
    return _LOGGING_ENABLED

def get_time():
    """
    Return the fixed time when logging was enabled.

    This function returns the timestamp when logging was first enabled. If logging hasn't
    been enabled yet, it returns None.

    Returns:
        datetime: The timestamp when logging was enabled, or None if logging hasn't been enabled.

    Example usage:
        import logger
        log_time = logger.get_time()
        if log_time:
            print(f"Logging was enabled at: {log_time}")
        else:
            print("Logging has not been enabled.")
    """
    if _CURRENT_TIME is None:
        return None  # Return None if logging hasn't been enabled
    return _CURRENT_TIME

