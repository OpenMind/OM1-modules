import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener


@dataclass
class LoggingConfig:
    """
    Logging configuration for the application.

    This class holds the configuration for logging, including the log level and whether to log to a file.

    Parameters
    ----------
    log_level : str
        The logging level to set. Defaults to "INFO".
        Can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    log_to_file : bool
        If True, log messages will also be written to a file. Defaults to False.
    """

    log_level: str = "INFO"
    log_to_file: bool = False


def setup_logging(
    config_name: str,
    log_level: str = "INFO",
    log_to_file: bool = False,
    logging_config: LoggingConfig = None,
) -> None:
    """
    Set up the logging configuration for the application.

    Parameters
    ----------
    config_name : str
        The name of the configuration file to use for logging.
    log_level : str, optional
        The logging level to set. Defaults to "INFO".
        Can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    log_to_file : bool, optional
        If True, log messages will also be written to a file.
        Defaults to False.
    logging_config : LoggingConfig, optional
        An optional LoggingConfig instance to use for logging configuration.
        If provided, it will override the `log_level` and `log_to_file` parameters.
    """
    if logging_config:
        log_level = logging_config.log_level
        log_to_file = logging_config.log_to_file

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.getLogger().handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    if log_to_file:
        os.makedirs("logs", exist_ok=True)

        file_handler = logging.FileHandler(
            f"logs/{config_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
            mode="a",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)


def get_logging_config() -> LoggingConfig:
    """
    Get the current logging configuration.

    Returns
    -------
    LoggingConfig
        The current logging configuration.
    """
    return LoggingConfig(
        log_level=logging.getLevelName(logging.getLogger().level),
        log_to_file=any(
            isinstance(handler, logging.FileHandler)
            for handler in logging.getLogger().handlers
        ),
    )


def setup_logging_mp_main(
    *,
    config_name: str,
    log_level: str = "INFO",
    log_to_file: bool = False,
    logging_config: LoggingConfig | None = None,
):
    """
    Configure logging via setup_logging(), then start a QueueListener so child
    processes can forward logs to the same handlers.

    Returns
    -------
    (listener, log_queue)
        listener : logging.handlers.QueueListener  (call .stop() on shutdown)
        log_queue : multiprocessing.Queue           (pass to child processes)
    """
    setup_logging(
        config_name=config_name,
        log_level=log_level,
        log_to_file=log_to_file,
        logging_config=logging_config,
    )

    root = logging.getLogger()
    handlers = list(root.handlers)
    log_queue: mp.Queue = mp.Queue(-1)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    logging.getLogger(__name__).info("QueueListener started for multiprocessing logs.")
    return listener, log_queue


def setup_logging_mp_child(log_queue: mp.Queue, *, level: str | int | None = None):
    """
    In a child process, route all logs to the main process using QueueHandler.

    Parameters
    ----------
    log_queue : multiprocessing.Queue
        The queue returned by setup_logging_mp_main().
    level : str | int, optional
        Optional override for root logger level in the child.
    """
    root = logging.getLogger()
    # Avoid duplicate/competing handlers in children
    for h in list(root.handlers):
        root.removeHandler(h)

    qh = QueueHandler(log_queue)
    root.addHandler(qh)

    if isinstance(level, str):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
    elif isinstance(level, int):
        root.setLevel(level)

    logging.getLogger(__name__).debug(
        "Child process logging configured (QueueHandler)."
    )
