"""Simple and elegant logging setup."""
import logging
from tqdm import tqdm

class Colors:
    """ANSI color codes for prettier output."""
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLUE = '\033[34m'
    GRAY = '\033[37m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

class Symbols:
    """Unicode symbols for status indicators."""
    CHECK = '✓'
    CROSS = '✗'
    ROCKET = '🚀'
    GEAR = '⚙️'
    PACKAGE = '📦'
    TRUCK = '🚛'
    CHART = '📊'

class SimpleFormatter(logging.Formatter):
    """Clean formatter with colors for better readability."""
    def format(self, record):
        color = {
            'DEBUG': Colors.GRAY,
            'INFO': Colors.CYAN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
            'CRITICAL': Colors.RED + Colors.BOLD
        }.get(record.levelname, Colors.RESET)
        
        # Use record.getMessage() to include formatting with args
        message = record.getMessage()
        return f"{color}{message}{Colors.RESET}"

def setup_logging():
    """Configure clean and simple logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(SimpleFormatter())
    logger.addHandler(console)

class ProgressTracker:
    """Simple progress tracking with tqdm."""
    def __init__(self, steps):
        self.steps = steps
        self.pbar = tqdm(
            total=len(steps), 
            desc=f"{Colors.BLUE}{Symbols.ROCKET} Optimization Progress{Colors.RESET}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        )
        self.current = 0
        
        # Status colors for different types of messages
        self.status_formats = {
            'success': f"{Colors.GREEN}{Symbols.CHECK}",
            'warning': f"{Colors.YELLOW}{Symbols.GEAR}",
            'error': f"{Colors.RED}{Symbols.CROSS}",
            'info': f"{Colors.CYAN}{Symbols.PACKAGE}",
        }
    
    def advance(self, message=None, status='success'):
        """Advance progress bar and optionally log a message."""
        if message:
            prefix = self.status_formats.get(status, '')
            formatted_message = f"{prefix} {message}{Colors.RESET}"
            self.pbar.write(formatted_message)
        self.current += 1
        self.pbar.update(1)
    
    def close(self):
        """Clean up progress bar."""
        self.pbar.write(f"\n{Colors.GREEN}{Symbols.ROCKET} Optimization completed!{Colors.RESET}\n")
        self.pbar.close()