"""
MedSync AI Configuration Settings
"""
import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    ENV_FILE = PROJECT_ROOT.parent / ".env"
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
except ImportError:
    pass

# Data Paths
DATA_DIR = PROJECT_ROOT.parent / "data"
MIMIC_DIR = DATA_DIR / "mimic-iv-clinical-database-demo-2.2"
ENDOVIS_DIR = DATA_DIR / "endovis2018"
MEDICAL_O1_PATH = DATA_DIR / "medical_o1_sft.json"

# ChromaDB Configuration
CHROMADB_DIR = PROJECT_ROOT / "chromadb_store"
CHROMADB_COLLECTION_CLINICAL = "clinical_knowledge"
CHROMADB_COLLECTION_SURGICAL = "surgical_reasoning"
CHROMADB_COLLECTION_VQLA = "vqla_labels"

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Uses sentence-transformers (accessible, no auth needed)
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# API Configuration Constants
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB for medical images

# LLM Configuration - Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Fast and multimodal production model
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096

# Patient Context Configuration (Side Panel)
DEFAULT_PATIENT_CONTEXT = {
    "vitals": {
        "blood_pressure": "",
        "blood_sugar": "",
        "weight": ""
    },
    "history": {
        "medications": [],
        "previous_surgeries": [],
        "allergies": []
    }
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Data Processing
BATCH_SIZE = 32
NUM_WORKERS = 4

# Risk Detection Thresholds
HEMORRHAGE_RISK_THRESHOLD = 0.7
ANESTHETIC_REACTION_THRESHOLD = 0.6
COMPLICATION_ALERT_THRESHOLD = 0.65

# VQLA Configuration
SURGICAL_INSTRUMENTS = [
    "Grasper", "Bipolar", "Hook", "Scissors", "Clipper",
    "Knot Pusher", "Suction", "Needle Driver"
]

ANATOMICAL_STRUCTURES = [
    "Gallbladder", "Liver", "Blood Vessel", "Peritoneum",
    "Stomach", "Bowel", "Omentum", "Ligament"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "medsync.log"

# Patient Linking
MIN_MATCHING_SCORE = 0.8
