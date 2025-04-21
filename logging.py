import logging
import os
import datetime
import sys


def setup_logging(log_file='simulation.log', console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Setup logging configuration for the simulation.

    Parameters:
        log_file: Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(file_level)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log start of session
    logging.info('-' * 80)
    logging.info(f'Logging session started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info('-' * 80)


def log_info(message):
    """Log an info message"""
    logging.info(message)


def log_debug(message):
    """Log a debug message"""
    logging.debug(message)


def log_warning(message):
    """Log a warning message"""
    logging.warning(message)


def log_error(message):
    """Log an error message"""
    logging.error(message)


def log_critical(message):
    """Log a critical message"""
    logging.critical(message)


def log_simulation_parameters(params):
    """
    Log the simulation parameters.

    Parameters:
        params: Dictionary of simulation parameters
    """
    logging.info("Simulation parameters:")
    for key, value in params.items():
        logging.info(f"  {key}: {value}")


def log_progress(current, total, message="Processing"):
    """
    Log progress of a long-running operation.

    Parameters:
        current: Current progress
        total: Total work to be done
        message: Message describing the operation
    """
    percent = (current / total) * 100
    logging.info(f"{message}: {current}/{total} ({percent:.1f}%)")


def log_result(name, value, unit=None):
    """
    Log a numerical result.

    Parameters:
        name: Name of the result
        value: Value of the result
        unit: Optional unit for the result
    """
    if unit:
        logging.info(f"Result - {name}: {value} {unit}")
    else:
        logging.info(f"Result - {name}: {value}")


def log_exception(exception, message="An exception occurred"):
    """
    Log an exception with traceback.

    Parameters:
        exception: The exception object
        message: Additional message to log with the exception
    """
    logging.exception(f"{message}: {str(exception)}")
