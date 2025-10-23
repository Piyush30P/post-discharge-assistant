"""
Logging utility for Post Discharge Medical AI Assistant
Provides comprehensive logging for all system operations
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import LOG_FILE, LOG_LEVEL


class SystemLogger:
    """Centralized logger for the entire system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger("PostDischargeAssistant")
        self.logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        LOG_FILE.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*80)
        self.logger.info("System Logger Initialized")
        self.logger.info("="*80)
    
    def log_agent_action(self, agent_name: str, action: str, details: Optional[str] = None):
        """Log agent actions"""
        msg = f"[{agent_name.upper()}] {action}"
        if details:
            msg += f" - {details}"
        self.logger.info(msg)
    
    def log_tool_call(self, tool_name: str, params: dict, result_summary: str):
        """Log tool invocations"""
        self.logger.info(f"[TOOL] {tool_name} called with params: {params}")
        self.logger.info(f"[TOOL] {tool_name} result: {result_summary}")
    
    def log_database_operation(self, operation: str, details: str):
        """Log database operations"""
        self.logger.info(f"[DATABASE] {operation}: {details}")
    
    def log_rag_query(self, query: str, num_results: int, sources: list):
        """Log RAG queries"""
        self.logger.info(f"[RAG] Query: {query}")
        self.logger.info(f"[RAG] Retrieved {num_results} chunks from sources: {sources}")
    
    def log_web_search(self, query: str, num_results: int):
        """Log web searches"""
        self.logger.info(f"[WEB_SEARCH] Query: {query}")
        self.logger.info(f"[WEB_SEARCH] Found {num_results} results")
    
    def log_patient_interaction(self, patient_name: str, query: str, agent: str):
        """Log patient interactions"""
        self.logger.info(f"[INTERACTION] Patient: {patient_name} | Agent: {agent}")
        self.logger.info(f"[INTERACTION] Query: {query}")
    
    def log_agent_handoff(self, from_agent: str, to_agent: str, reason: str):
        """Log agent transitions"""
        self.logger.info(f"[HANDOFF] {from_agent} → {to_agent}")
        self.logger.info(f"[HANDOFF] Reason: {reason}")
    
    def log_error(self, component: str, error: Exception):
        """Log errors with full traceback"""
        self.logger.error(f"[ERROR] {component}: {str(error)}", exc_info=True)
    
    def info(self, message: str):
        """General info logging"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Debug logging"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Warning logging"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Error logging"""
        self.logger.error(message)


# Global logger instance
logger = SystemLogger()


def get_logger():
    """Get the global logger instance"""
    return logger


if __name__ == "__main__":
    # Test logging
    logger = get_logger()
    logger.info("Logger test - Info message")
    logger.debug("Logger test - Debug message")
    logger.log_agent_action("Receptionist", "Patient lookup", "John Smith")
    logger.log_tool_call("get_patient_data", {"name": "John Smith"}, "Patient found")
    print(f"\nLog file created at: {LOG_FILE}")