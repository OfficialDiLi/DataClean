"""Utility functions for logging."""

import logging
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "app.log"), logging.StreamHandler()],
)

logger = logging.getLogger("DataClean")
