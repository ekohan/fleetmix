import logging

import pytest

from fleetmix.utils.logging import Colors, ProgressTracker, SimpleFormatter


class DummyRecord(logging.LogRecord):
    def __init__(self, levelname, msg):
        super().__init__(
            name="test",
            level=getattr(logging, levelname),
            pathname=__file__,
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
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


@pytest.mark.parametrize(
    "level, color",
    [
        ("DEBUG", Colors.GRAY),
        ("INFO", Colors.CYAN),
        ("WARNING", Colors.YELLOW),
        ("ERROR", Colors.RED),
        ("CRITICAL", Colors.RED + Colors.BOLD),
    ],
)
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
    monkeypatch.setattr(logging_utils, "tqdm", lambda total, desc, bar_format: dummy)

    steps = ["a", "b", "c"]
    pt = ProgressTracker(steps)

    # Advance with message
    pt.advance("msg1", status="success")
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
