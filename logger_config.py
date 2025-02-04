import logging
from pathlib import Path
from datetime import datetime

class LoggerConfig:
    """Configure logging with file and console handlers"""
    
    def __init__(self, log_dir:str = "logs"):
        """Initialize logger configuration"""
        """
        Args:
            log_dir (str) : Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.setup_log_directory()
        
    def setup_log_directory(self):
        """Create log directory if doesn't exist"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def get_logger(self, logger_name:str) -> logging.Logger:
        """Get configured logger instance

        Args:
            logger_name (str): Name of the logger

        Returns:
            logging.Logger: Configured logger instance
        """
        # Create or retrieve logger
        logger = logging.getLogger(logger_name)
        # Set log level to minimum severity level, INFO,
        # logging.INFO means the logger will handle messages at INFO level and above (i.e., INFO, WARNING, ERROR, CRITICAL), but not DEBUG messages.
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
            
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Create daily log file handler
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"{logger_name}_{current_date}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_error_logger(self, logger_name: str) -> logging.Logger:
        """
        Get a logger specifically for error tracking
        
        Args:
            logger_name (str): Name of the logger
            
        Returns:
            logging.Logger: Configured error logger instance
        """
        # Create error logger
        error_logger = logging.getLogger(f"{logger_name}_error")
        error_logger.setLevel(logging.ERROR)
        
        # Remove existing handlers if any
        if error_logger.handlers:
            error_logger.handlers.clear()
        
        # Create formatter with detailed error information
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\nStack Trace: %(exc_info)s')
        
        # Create error log file handler
        error_log_file = self.log_dir / f"{logger_name}_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(error_formatter)
        
        # Add handler to logger
        error_logger.addHandler(error_handler)
        
        return error_logger
        