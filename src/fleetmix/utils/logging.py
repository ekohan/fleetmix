"""Simple and elegant logging setup with verbosity level management."""

import logging
import os
from enum import Enum

from tqdm import tqdm


class LogLevel(Enum):
    """Verbosity levels for Fleetmix logging."""

    QUIET = 0  # Errors only
    NORMAL = 1  # Progress indication (default)
    VERBOSE = 2  # Detailed progress
    DEBUG = 3  # Full diagnostics


class Colors:
    """ANSI color codes for prettier output."""

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GRAY = "\033[37m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class Symbols:
    """Unicode symbols for status indicators."""

    CHECK = "✓"
    CROSS = "✗"
    ROCKET = "🚀"
    GEAR = "⚙️"
    PACKAGE = "📦"
    TRUCK = "🚛"
    CHART = "📊"
    INFO = "ℹ️"
    WARNING = "⚠️"
    SUCCESS = "✅"


class SimpleFormatter(logging.Formatter):
    """Clean formatter with colors for better readability."""

    def format(self, record):
        color = {
            "DEBUG": Colors.GRAY,
            "INFO": Colors.CYAN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED,
            "CRITICAL": Colors.RED + Colors.BOLD,
        }.get(record.levelname, Colors.RESET)

        # Use record.getMessage() to include formatting with args
        message = record.getMessage()
        return f"{color}{message}{Colors.RESET}"


class FleetmixLogger:
    """Centralized logger for Fleetmix with level management."""

    _current_level = LogLevel.NORMAL
    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def set_level(cls, level: LogLevel):
        """Set the global logging level for all Fleetmix loggers."""
        cls._current_level = level

        # Update all existing loggers
        for logger_instance in cls._loggers.values():
            cls._configure_logger_level(logger_instance, level)

    @classmethod
    def get_level(cls) -> LogLevel:
        """Get the current logging level."""
        return cls._current_level

    @classmethod
    def _configure_logger_level(cls, logger: logging.Logger, level_param: LogLevel):
        """Configure logger level based on FleetmixLogger level_param."""

        effective_loglevel = level_param

        # Check environment variable for globally set level (e.g., from main process)
        env_level_name = os.getenv("FLEETMIX_EFFECTIVE_LOG_LEVEL")
        if env_level_name:
            try:
                level_from_env = LogLevel[env_level_name]
                # If level_param is the default (likely in a new subprocess inheriting default _current_level),
                # and env var specifies a non-default level, prioritize env var.
                if level_param == LogLevel.NORMAL and level_from_env != LogLevel.NORMAL:
                    effective_loglevel = level_from_env
                # If level_param is already non-NORMAL (e.g. set_level was called explicitly in this process),
                # it might take precedence, or we might always sync with env var if it's considered authoritative.
                # For now, this logic prioritizes env var if current is default.
            except KeyError:
                # Invalid value in env var, ignore
                pass

        if effective_loglevel == LogLevel.QUIET:
            logger.setLevel(logging.ERROR)
        elif effective_loglevel == LogLevel.NORMAL:
            logger.setLevel(logging.INFO)
        elif effective_loglevel == LogLevel.VERBOSE:
            logger.setLevel(logging.INFO)  # Same as normal, but affects output content
        elif effective_loglevel == LogLevel.DEBUG:
            logger.setLevel(logging.DEBUG)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a configured logger for the given module."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._configure_logger_level(logger, cls._current_level)
            cls._loggers[name] = logger
        return cls._loggers[name]

    @classmethod
    def progress(cls, message: str, symbol: str = Symbols.GEAR):
        """Log a progress message (shown in NORMAL and above)."""
        if cls._current_level.value >= LogLevel.NORMAL.value:
            logger = cls.get_logger("fleetmix.progress")
            logger.info(f"{symbol} {message}")

    @classmethod
    def success(cls, message: str, symbol: str = Symbols.CHECK):
        """Log a success message (shown in NORMAL and above)."""
        if cls._current_level.value >= LogLevel.NORMAL.value:
            logger = cls.get_logger("fleetmix.progress")
            logger.info(f"{Colors.GREEN}{symbol} {message}{Colors.RESET}")

    @classmethod
    def detail(cls, message: str, symbol: str = "  "):
        """Log a detailed message (shown in VERBOSE and above)."""
        if cls._current_level.value >= LogLevel.VERBOSE.value:
            logger = cls.get_logger("fleetmix.detail")
            logger.info(f"{symbol} {message}")

    @classmethod
    def debug(cls, message: str, logger_name: str = "fleetmix.debug"):
        """Log a debug message (shown in DEBUG only)."""
        if cls._current_level.value >= LogLevel.DEBUG.value:
            logger = cls.get_logger(logger_name)
            logger.debug(message)

    @classmethod
    def warning(cls, message: str, symbol: str = Symbols.WARNING):
        """Log a warning message (shown in all levels except QUIET)."""
        if cls._current_level.value >= LogLevel.NORMAL.value:
            logger = cls.get_logger("fleetmix.warning")
            logger.warning(f"{symbol} {message}")

    @classmethod
    def error(cls, message: str, symbol: str = Symbols.CROSS):
        """Log an error message (shown in all levels)."""
        logger = cls.get_logger("fleetmix.error")
        logger.error(f"{symbol} {message}")


def suppress_third_party_logs():
    """Suppress noisy third-party library logs."""
    # Suppress Numba debug output
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)

    # Suppress other common noisy loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def setup_logging(level: LogLevel | None = None):
    """Configure clean and simple logging with optional level override."""
    # Determine log level from various sources
    if level is None:
        # Check environment variable
        env_level = os.getenv("FLEETMIX_LOG_LEVEL", "").lower()
        if env_level == "quiet":
            level = LogLevel.QUIET
        elif env_level == "verbose":
            level = LogLevel.VERBOSE
        elif env_level == "debug":
            level = LogLevel.DEBUG
        else:
            level = LogLevel.NORMAL

    # Set the global level
    FleetmixLogger.set_level(level)
    # Store the determined level in an environment variable for subprocesses
    os.environ["FLEETMIX_EFFECTIVE_LOG_LEVEL"] = level.name

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Allow all messages, filter at handler level

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler
    console = logging.StreamHandler()
    console.setFormatter(SimpleFormatter())

    # Set the handler's level based on the desired Fleetmix log level
    if level == LogLevel.QUIET:
        console.setLevel(logging.ERROR)
    elif level == LogLevel.NORMAL:
        console.setLevel(logging.INFO)
    elif level == LogLevel.VERBOSE:
        console.setLevel(
            logging.INFO
        )  # Or logging.DEBUG if VERBOSE means more than INFO
    elif level == LogLevel.DEBUG:
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

    # Suppress third-party noise
    suppress_third_party_logs()


class ProgressTracker:
    """Simple progress tracking with tqdm."""

    def __init__(self, steps):
        self.steps = steps
        self.show_progress = FleetmixLogger.get_level().value >= LogLevel.NORMAL.value

        self.pbar: tqdm | None
        if self.show_progress:
            self.pbar = tqdm(
                total=len(steps),
                desc=f"{Colors.BLUE}{Symbols.ROCKET} Optimization Progress{Colors.RESET}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )
        else:
            self.pbar = None
        self.current = 0

        # Status colors for different types of messages
        self.status_formats = {
            "success": f"{Colors.GREEN}{Symbols.CHECK}",
            "warning": f"{Colors.YELLOW}{Symbols.GEAR}",
            "error": f"{Colors.RED}{Symbols.CROSS}",
            "info": f"{Colors.CYAN}{Symbols.PACKAGE}",
        }

    def advance(self, message=None, status="success"):
        """Advance progress bar and optionally log a message."""
        if self.show_progress and self.pbar:
            if message:
                prefix = self.status_formats.get(status, "")
                formatted_message = f"{prefix} {message}{Colors.RESET}"
                self.pbar.write(formatted_message)
            self.current += 1
            self.pbar.update(1)

    def close(self):
        """Clean up progress bar."""
        if self.show_progress and self.pbar:
            self.pbar.write(
                f"\n{Colors.GREEN}{Symbols.ROCKET} Optimization completed!{Colors.RESET}\n"
            )
            self.pbar.close()


# Convenience functions for common logging patterns
def log_progress(message: str, symbol: str = Symbols.GEAR):
    """Log a progress message."""
    FleetmixLogger.progress(message, symbol)


def log_success(message: str, symbol: str = Symbols.CHECK):
    """Log a success message."""
    FleetmixLogger.success(message, symbol)


def log_detail(message: str, symbol: str = "  "):
    """Log a detailed message."""
    FleetmixLogger.detail(message, symbol)


def log_warning(message: str, symbol: str = Symbols.WARNING):
    """Log a warning message."""
    FleetmixLogger.warning(message, symbol)


def log_error(message: str, symbol: str = Symbols.CROSS):
    """Log an error message."""
    FleetmixLogger.error(message, symbol)


def log_debug(message: str, logger_name: str = "fleetmix.debug"):
    """Log a debug message."""
    FleetmixLogger.debug(message, logger_name)
