import sys
import os
from loguru import logger
from pathlib import Path

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "human",
    log_dir: str = "logs",
    enable_file_logging: bool = False
):
    """
    Configure logging for the application.

    Args:
        log_level (str): The minimum log level to capture (e.g., "INFO", "DEBUG").
        log_format (str): The format for logs ('human' or 'json').
        log_dir (str): Directory to store log files if file logging is enabled.
        enable_file_logging (bool): If True, logs will be written to a file.
    """
    logger.remove()  # Remove default handler

    if log_format.lower() == "json":
        # Structured JSON logging to stdout, ideal for containers
        logger.add(
            sys.stdout,
            level=log_level.upper(),
            format="{message}",  # The message will be the JSON record
            serialize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Human-readable, colored logging to stderr for local development
        logger.add(
            sys.stderr,
            level=log_level.upper(),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            colorize=True,
        )

    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / "app_{time:YYYY-MM-DD}.log"

        logger.add(
            file_path,
            rotation="10 MB",
            retention="30 days",
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )

    logger.info(
        "Logging configured",
        level=log_level,
        format=log_format,
        file_logging=enable_file_logging
    )

    return logger

# Configure logger based on environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "human") # 'human' or 'json'
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"

# Initialize and export the logger instance
log = setup_logging(
    log_level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    enable_file_logging=ENABLE_FILE_LOGGING
)

