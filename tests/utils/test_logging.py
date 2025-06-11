"""Unit tests for the logging module."""

import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from fleetmix.utils.logging import (
    Colors,
    FleetmixLogger,
    LogLevel,
    ProgressTracker,
    SimpleFormatter,
    Symbols,
    log_debug,
    log_detail,
    log_error,
    log_progress,
    log_success,
    log_warning,
    setup_logging,
    suppress_third_party_logs,
)


class TestLogLevel(unittest.TestCase):
    """Test cases for LogLevel enum."""

    def test_log_levels(self):
        """Test that log levels have correct values."""
        self.assertEqual(LogLevel.QUIET.value, 0)
        self.assertEqual(LogLevel.NORMAL.value, 1)
        self.assertEqual(LogLevel.VERBOSE.value, 2)
        self.assertEqual(LogLevel.DEBUG.value, 3)


class TestColors(unittest.TestCase):
    """Test cases for Colors class."""

    def test_color_codes(self):
        """Test that color codes are defined."""
        self.assertEqual(Colors.CYAN, "\033[36m")
        self.assertEqual(Colors.GREEN, "\033[32m")
        self.assertEqual(Colors.RESET, "\033[0m")


class TestSymbols(unittest.TestCase):
    """Test cases for Symbols class."""

    def test_symbols(self):
        """Test that symbols are defined."""
        self.assertEqual(Symbols.CHECK, "✓")
        self.assertEqual(Symbols.CROSS, "✗")
        self.assertEqual(Symbols.ROCKET, "🚀")


class TestSimpleFormatter(unittest.TestCase):
    """Test cases for SimpleFormatter class."""

    def test_format_with_colors(self):
        """Test that formatter adds colors based on log level."""
        formatter = SimpleFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        self.assertIn(Colors.CYAN, formatted)
        self.assertIn("Test message", formatted)
        self.assertIn(Colors.RESET, formatted)

    def test_format_with_args(self):
        """Test that formatter handles message arguments."""
        formatter = SimpleFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Test %s message",
            args=("warning",),
            exc_info=None,
        )

        formatted = formatter.format(record)
        self.assertIn(Colors.YELLOW, formatted)
        self.assertIn("Test warning message", formatted)


class TestFleetmixLogger(unittest.TestCase):
    """Test cases for FleetmixLogger class."""

    def setUp(self):
        """Reset logger state before each test."""
        FleetmixLogger._current_level = LogLevel.NORMAL
        FleetmixLogger._loggers.clear()
        # Clear environment variable
        if "FLEETMIX_EFFECTIVE_LOG_LEVEL" in os.environ:
            del os.environ["FLEETMIX_EFFECTIVE_LOG_LEVEL"]

    def test_set_and_get_level(self):
        """Test setting and getting log level."""
        FleetmixLogger.set_level(LogLevel.DEBUG)
        self.assertEqual(FleetmixLogger.get_level(), LogLevel.DEBUG)

        FleetmixLogger.set_level(LogLevel.QUIET)
        self.assertEqual(FleetmixLogger.get_level(), LogLevel.QUIET)

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = FleetmixLogger.get_logger("test.module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test.module")

        # Getting same logger should return same instance
        logger2 = FleetmixLogger.get_logger("test.module")
        self.assertIs(logger, logger2)

    def test_configure_logger_level(self):
        """Test that logger level is configured based on FleetmixLogger level."""
        FleetmixLogger.set_level(LogLevel.QUIET)
        logger = FleetmixLogger.get_logger("test.quiet")
        self.assertEqual(logger.level, logging.ERROR)

        FleetmixLogger.set_level(LogLevel.DEBUG)
        logger = FleetmixLogger.get_logger("test.debug")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_environment_variable_override(self):
        """Test that environment variable can override log level."""
        # Set environment variable
        os.environ["FLEETMIX_EFFECTIVE_LOG_LEVEL"] = "DEBUG"

        # Create logger with NORMAL level
        FleetmixLogger.set_level(LogLevel.NORMAL)
        logger = FleetmixLogger.get_logger("test.env")

        # Should use DEBUG from environment
        self.assertEqual(logger.level, logging.DEBUG)

    @patch("logging.Logger.info")
    def test_progress_message(self, mock_info):
        """Test progress message logging."""
        FleetmixLogger.set_level(LogLevel.NORMAL)
        FleetmixLogger.progress("Test progress")

        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        self.assertIn("Test progress", call_args)
        self.assertIn(Symbols.GEAR, call_args)

    @patch("logging.Logger.info")
    def test_success_message(self, mock_info):
        """Test success message logging."""
        FleetmixLogger.set_level(LogLevel.NORMAL)
        FleetmixLogger.success("Test success")

        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        self.assertIn("Test success", call_args)
        self.assertIn(Colors.GREEN, call_args)

    @patch("logging.Logger.info")
    def test_detail_message_verbose_only(self, mock_info):
        """Test that detail messages only show in VERBOSE mode."""
        # Should not log in NORMAL mode
        FleetmixLogger.set_level(LogLevel.NORMAL)
        FleetmixLogger.detail("Test detail")
        mock_info.assert_not_called()

        # Should log in VERBOSE mode
        FleetmixLogger.set_level(LogLevel.VERBOSE)
        FleetmixLogger.detail("Test detail")
        mock_info.assert_called_once()

    @patch("logging.Logger.debug")
    def test_debug_message_debug_only(self, mock_debug):
        """Test that debug messages only show in DEBUG mode."""
        # Should not log in VERBOSE mode
        FleetmixLogger.set_level(LogLevel.VERBOSE)
        FleetmixLogger.debug("Test debug")
        mock_debug.assert_not_called()

        # Should log in DEBUG mode
        FleetmixLogger.set_level(LogLevel.DEBUG)
        FleetmixLogger.debug("Test debug")
        mock_debug.assert_called_once()

    @patch("logging.Logger.warning")
    def test_warning_message(self, mock_warning):
        """Test warning message logging."""
        FleetmixLogger.set_level(LogLevel.NORMAL)
        FleetmixLogger.warning("Test warning")

        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        self.assertIn("Test warning", call_args)
        self.assertIn(Symbols.WARNING, call_args)

    @patch("logging.Logger.error")
    def test_error_message(self, mock_error):
        """Test error message logging."""
        # Should log even in QUIET mode
        FleetmixLogger.set_level(LogLevel.QUIET)
        FleetmixLogger.error("Test error")

        mock_error.assert_called_once()
        call_args = mock_error.call_args[0][0]
        self.assertIn("Test error", call_args)
        self.assertIn(Symbols.CROSS, call_args)


class TestLoggingSetup(unittest.TestCase):
    """Test cases for logging setup functions."""

    def test_suppress_third_party_logs(self):
        """Test that third-party loggers are suppressed."""
        suppress_third_party_logs()

        # Check that loggers are set to WARNING level
        self.assertEqual(logging.getLogger("numba").level, logging.WARNING)
        self.assertEqual(logging.getLogger("matplotlib").level, logging.WARNING)
        self.assertEqual(logging.getLogger("urllib3").level, logging.WARNING)

    @patch.dict(os.environ, {}, clear=True)
    def test_setup_logging_default(self):
        """Test setup_logging with default level."""
        setup_logging()
        self.assertEqual(FleetmixLogger.get_level(), LogLevel.NORMAL)
        self.assertEqual(os.environ["FLEETMIX_EFFECTIVE_LOG_LEVEL"], "NORMAL")

    @patch.dict(os.environ, {"FLEETMIX_LOG_LEVEL": "debug"}, clear=True)
    def test_setup_logging_from_env(self):
        """Test setup_logging reads from environment variable."""
        setup_logging()
        self.assertEqual(FleetmixLogger.get_level(), LogLevel.DEBUG)

    def test_setup_logging_with_explicit_level(self):
        """Test setup_logging with explicit level parameter."""
        setup_logging(LogLevel.QUIET)
        self.assertEqual(FleetmixLogger.get_level(), LogLevel.QUIET)


class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker class."""

    @patch("fleetmix.utils.logging.tqdm")
    def test_progress_tracker_normal_mode(self, mock_tqdm):
        """Test ProgressTracker in NORMAL mode."""
        FleetmixLogger.set_level(LogLevel.NORMAL)

        steps = ["step1", "step2", "step3"]
        tracker = ProgressTracker(steps)

        # Should create progress bar
        mock_tqdm.assert_called_once()
        self.assertIsNotNone(tracker.pbar)

    def test_progress_tracker_quiet_mode(self):
        """Test ProgressTracker in QUIET mode."""
        FleetmixLogger.set_level(LogLevel.QUIET)

        steps = ["step1", "step2"]
        tracker = ProgressTracker(steps)

        # Should not create progress bar
        self.assertIsNone(tracker.pbar)

    @patch("fleetmix.utils.logging.tqdm")
    def test_progress_tracker_advance(self, mock_tqdm):
        """Test advancing progress tracker."""
        FleetmixLogger.set_level(LogLevel.NORMAL)
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = ProgressTracker(["step1"])
        tracker.advance("Completed step", status="success")

        # Should write message and update progress
        mock_pbar.write.assert_called_once()
        mock_pbar.update.assert_called_once_with(1)

    @patch("fleetmix.utils.logging.tqdm")
    def test_progress_tracker_close(self, mock_tqdm):
        """Test closing progress tracker."""
        FleetmixLogger.set_level(LogLevel.NORMAL)
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = ProgressTracker(["step1"])
        tracker.close()

        # Should write completion message and close
        mock_pbar.write.assert_called()
        mock_pbar.close.assert_called_once()


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience logging functions."""

    @patch.object(FleetmixLogger, "progress")
    def test_log_progress(self, mock_progress):
        """Test log_progress convenience function."""
        log_progress("Test message")
        mock_progress.assert_called_once_with("Test message", Symbols.GEAR)

    @patch.object(FleetmixLogger, "success")
    def test_log_success(self, mock_success):
        """Test log_success convenience function."""
        log_success("Test message")
        mock_success.assert_called_once_with("Test message", Symbols.CHECK)

    @patch.object(FleetmixLogger, "detail")
    def test_log_detail(self, mock_detail):
        """Test log_detail convenience function."""
        log_detail("Test message")
        mock_detail.assert_called_once_with("Test message", "  ")

    @patch.object(FleetmixLogger, "warning")
    def test_log_warning(self, mock_warning):
        """Test log_warning convenience function."""
        log_warning("Test message")
        mock_warning.assert_called_once_with("Test message", Symbols.WARNING)

    @patch.object(FleetmixLogger, "error")
    def test_log_error(self, mock_error):
        """Test log_error convenience function."""
        log_error("Test message")
        mock_error.assert_called_once_with("Test message", Symbols.CROSS)

    @patch.object(FleetmixLogger, "debug")
    def test_log_debug(self, mock_debug):
        """Test log_debug convenience function."""
        log_debug("Test message", "custom.logger")
        mock_debug.assert_called_once_with("Test message", "custom.logger")


if __name__ == "__main__":
    unittest.main()
