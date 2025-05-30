import logging
import pytest
from fleetmix.utils.logging import SimpleFormatter, ProgressTracker, Colors
import os
from unittest.mock import patch, MagicMock, call
from fleetmix.utils.logging import (
    LogLevel, FleetmixLogger, setup_logging, ProgressTracker, 
    log_detail, log_debug
)

class DummyRecord(logging.LogRecord):
    def __init__(self, levelname, msg):
        super().__init__(name="test", level=getattr(logging, levelname), pathname=__file__, lineno=0, msg=msg, args=(), exc_info=None)
        self.levelname = levelname

class DummyBar:
    def __init__(self):
        self.updates = []
        self.writes = []
        self.closed = False
    def update(self, n):
        self.updates.append(n)
    def write(self, msg):
        self.writes.append(msg)
    def close(self):
        self.closed = True

@pytest.mark.parametrize("level, color", [
    ("DEBUG", Colors.GRAY),
    ("INFO", Colors.CYAN),
    ("WARNING", Colors.YELLOW),
    ("ERROR", Colors.RED),
    ("CRITICAL", Colors.RED + Colors.BOLD)
])
def test_simple_formatter_colors(level, color):
    fmt = SimpleFormatter()
    rec = DummyRecord(level, "hello")
    out = fmt.format(rec)
    assert out.startswith(color)
    assert out.endswith(Colors.RESET)
    assert "hello" in out


def test_progress_tracker_advance_and_close(monkeypatch):
    # Monkeypatch tqdm to return our DummyBar
    import fleetmix.utils.logging as logging_utils
    dummy = DummyBar()
    monkeypatch.setattr(logging_utils, 'tqdm', lambda total, desc, bar_format: dummy)

    steps = ['a', 'b', 'c']
    pt = ProgressTracker(steps)

    # Advance with message
    pt.advance("msg1", status='success')
    # Advance without message
    pt.advance()
    # Close
    pt.close()

    # After two advances, updates should be [1,1]
    assert dummy.updates == [1, 1]
    # Write called once for message and once on close
    assert any("msg1" in w for w in dummy.writes)
    assert any("completed" in w.lower() for w in dummy.writes)
    # Close should mark closed True
    assert dummy.closed 


class TestFleetmixLogger:
    """Test FleetmixLogger class methods"""
    
    def test_configure_logger_level_invalid_env_var(self):
        """Test _configure_logger_level handles invalid FLEETMIX_EFFECTIVE_LOG_LEVEL gracefully"""
        with patch.dict(os.environ, {'FLEETMIX_EFFECTIVE_LOG_LEVEL': 'INVALID_LEVEL'}):
            logger = MagicMock()
            # Should not raise exception and should use the passed level
            FleetmixLogger._configure_logger_level(logger, LogLevel.VERBOSE)
            logger.setLevel.assert_called_with(logging.INFO)
    
    def test_detail_method_verbose_level(self):
        """Test detail method logs when level is VERBOSE"""
        FleetmixLogger.set_level(LogLevel.VERBOSE)
        with patch.object(FleetmixLogger, 'get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            FleetmixLogger.detail("Test detail message")
            
            mock_get_logger.assert_called_with('fleetmix.detail')
            mock_logger.info.assert_called_once_with("   Test detail message")
    
    def test_debug_method_debug_level(self):
        """Test debug method logs when level is DEBUG"""
        FleetmixLogger.set_level(LogLevel.DEBUG)
        with patch.object(FleetmixLogger, 'get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            FleetmixLogger.debug("Test debug message", "custom.logger")
            
            mock_get_logger.assert_called_with('custom.logger')
            mock_logger.debug.assert_called_once_with("Test debug message")


class TestSetupLogging:
    """Test setup_logging function"""
    
    def test_setup_logging_env_var_quiet(self):
        """Test setup_logging with FLEETMIX_LOG_LEVEL=quiet"""
        with patch.dict(os.environ, {'FLEETMIX_LOG_LEVEL': 'quiet'}):
            with patch.object(FleetmixLogger, 'set_level') as mock_set_level:
                setup_logging()
                mock_set_level.assert_called_with(LogLevel.QUIET)
    
    def test_setup_logging_env_var_verbose(self):
        """Test setup_logging with FLEETMIX_LOG_LEVEL=verbose"""
        with patch.dict(os.environ, {'FLEETMIX_LOG_LEVEL': 'verbose'}):
            with patch.object(FleetmixLogger, 'set_level') as mock_set_level:
                setup_logging()
                mock_set_level.assert_called_with(LogLevel.VERBOSE)
    
    def test_setup_logging_env_var_debug(self):
        """Test setup_logging with FLEETMIX_LOG_LEVEL=debug"""
        with patch.dict(os.environ, {'FLEETMIX_LOG_LEVEL': 'debug'}):
            with patch.object(FleetmixLogger, 'set_level') as mock_set_level:
                setup_logging()
                mock_set_level.assert_called_with(LogLevel.DEBUG)
    
    def test_setup_logging_default_case(self):
        """Test setup_logging default case in handler level setting"""
        # Create a custom LogLevel that's not handled by specific cases
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_handler = MagicMock()
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger
            
            with patch('logging.StreamHandler') as mock_stream_handler:
                mock_stream_handler.return_value = mock_handler
                
                # Set an unexpected level value to trigger default case
                with patch.object(FleetmixLogger, '_current_level', 99):  # Invalid level
                    setup_logging(level=LogLevel.NORMAL)
                    
                # Verify handler was configured
                mock_logger.addHandler.assert_called()

    def test_setup_logging_unhandled_level(self):
        """Test setup_logging with an unhandled log level to trigger else case"""
        # Create a mock LogLevel that's not one of the handled cases
        mock_level = MagicMock()
        mock_level.name = "UNHANDLED"
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_handler = MagicMock()
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger
            
            with patch('logging.StreamHandler') as mock_stream_handler:
                mock_stream_handler.return_value = mock_handler
                
                # Pass the unhandled level
                setup_logging(level=mock_level)
                
                # The else case should set INFO level
                mock_handler.setLevel.assert_called_with(logging.INFO)


class TestProgressTracker:
    """Test ProgressTracker class"""
    
    def test_progress_tracker_quiet_mode(self):
        """Test ProgressTracker with QUIET log level (no progress bar)"""
        FleetmixLogger.set_level(LogLevel.QUIET)
        
        tracker = ProgressTracker(['step1', 'step2'])
        assert tracker.pbar is None
        assert tracker.show_progress is False
        
        # Should not raise errors even without progress bar
        tracker.advance("Test message")
        tracker.close()


class TestConvenienceFunctions:
    """Test convenience logging functions"""
    
    def test_log_detail(self):
        """Test log_detail convenience function"""
        with patch.object(FleetmixLogger, 'detail') as mock_detail:
            log_detail("Test message", ">>")
            mock_detail.assert_called_once_with("Test message", ">>")
    
    def test_log_debug(self):
        """Test log_debug convenience function"""
        with patch.object(FleetmixLogger, 'debug') as mock_debug:
            log_debug("Debug message", "test.logger")
            mock_debug.assert_called_once_with("Debug message", "test.logger") 