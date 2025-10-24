"""
Configuration module for Post Discharge Medical AI Assistant
Loads environment variables and provides centralized config access
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
VECTOR_STORE_DIR = BASE_DIR / "vectorstore"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# API Keys


# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # ADD THIS LINE
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nephrology-knowledge")
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / os.getenv("LOG_FILE", "system_logs.txt")

# Database
DATABASE_PATH = DATA_DIR / os.getenv("DATABASE_PATH", "patients.db").split("/")[-1]

# Vector Store
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# PDF Processing
NEPHROLOGY_PDF_PATH = DATA_DIR / os.getenv("NEPHROLOGY_PDF_PATH", "nephrology_book.pdf").split("/")[-1]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Model Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
MODEL_NAME = "gemini-pro"

# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚕️ **IMPORTANT MEDICAL DISCLAIMER**
This is an AI assistant for educational purposes only. 
Always consult healthcare professionals for medical advice.
This system does not replace professional medical consultation.
"""

def validate_config():
    """Validate that all required configurations are set"""
    errors = []
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        errors.append(
            "GOOGLE_API_KEY not set. Please set it in .env file.\n"
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        errors.append(
            "PINECONE_API_KEY not set. Please set it in .env file.\n"
            "Get your API key from: https://app.pinecone.io/"
        )
    
    if errors:
        raise ValueError("\n\n".join(errors))
    
    print("✓ Configuration validated successfully")
    return True

if __name__ == "__main__":
    print("Configuration Settings:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Logs Directory: {LOGS_DIR}")
    print(f"  Vector Store: {VECTOR_STORE_PATH}")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  LLM Model: {MODEL_NAME}")
    print(f"  Pinecone Index: {PINECONE_INDEX_NAME}")
    validate_config()