import logging

# Configure advanced logging
class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for better log visualization"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if record.levelname in self.COLORS:
            # Padded for alignment
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname:<8}{self.COLORS['RESET']}"
        else:
            record.levelname = f"{record.levelname:<8}" # Padded for alignment
        
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

# Setup logging
def get_logger(name: str = 'FluentCompute', level: int = logging.INFO) -> logging.Logger:
    """Gets a configured logger instance."""
    _logger = logging.getLogger(name)
    if not _logger.handlers: # Avoid adding multiple handlers if called multiple times
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(ColoredFormatter())
        _logger.addHandler(log_handler)
        _logger.setLevel(level)
        _logger.propagate = False # Avoid duplicate logs if root logger is configured
    else: # If already configured, just ensure level is set
        _logger.setLevel(level)
    return _logger

# Global logger instance (can be imported by other modules)
# Initialized with a default level, can be updated.
logger = get_logger()
