"""Configuration settings for the application."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Database
DB_NAME = os.getenv("DB_NAME", "users.db")
DB_PATH = BASE_DIR / DB_NAME

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
MAX_LOGIN_ATTEMPTS = 3
LOGIN_COOLDOWN = 300  # 5 minutes

# Cache settings
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 100  # Maximum number of items to cache

# File upload settings
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".ods"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
