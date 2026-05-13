"""
Logging utility for MedSync AI
"""
import logging
import sys
from pathlib import Path
from medsync_ai.config.settings import LOG_LEVEL, LOG_FILE, PROJECT_ROOT

# Create logs directory
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
