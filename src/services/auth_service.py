"""Authentication service for managing user operations."""

from typing import Optional
import bcrypt
import sqlite3
from src.config.config import DB_PATH
from src.utils.logging_utils import logger


class AuthService:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL,
                        login_attempts INTEGER DEFAULT 0,
                        last_attempt TIMESTAMP
                    )
                """
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def register(self, username: str, password: str) -> bool:
        """Register a new user."""
        if not username or not password:
            return False

        try:
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed_password),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Registration failed: Username {username} already exists")
            return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def login(self, username: str, password: str) -> bool:
        """Authenticate a user."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username = ?", (username,))
                result = c.fetchone()

                if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
                    # Reset login attempts on successful login
                    c.execute(
                        "UPDATE users SET login_attempts = 0 WHERE username = ?",
                        (username,),
                    )
                    conn.commit()
                    return True

                # Increment login attempts on failure
                c.execute(
                    """
                    UPDATE users 
                    SET login_attempts = login_attempts + 1,
                        last_attempt = CURRENT_TIMESTAMP
                    WHERE username = ?
                """,
                    (username,),
                )
                conn.commit()
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
