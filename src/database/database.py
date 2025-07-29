"""Database manager for the application."""

import sqlite3
from contextlib import contextmanager
from src.config.config import DB_PATH
from src.utils.logging_utils import logger


class DatabaseManager:
    """A manager for database operations."""

    def __init__(self, db_path: str):
        """Initialize the database manager."""
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they do not exist and run migrations."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._create_tables(cursor)
                self._run_migrations(cursor)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _create_tables(self, cursor):
        """Define and create the necessary tables."""
        # User table for authentication and tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
            """
        )

        # Session table for managing user sessions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (username) REFERENCES users (username)
            )
            """
        )

        # Activity log for tracking user actions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                activity TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username)
            )
            """
        )

    def _run_migrations(self, cursor):
        """Apply database schema migrations to the users table."""
        self._add_column_if_not_exists(cursor, "users", "login_attempts", "INTEGER DEFAULT 0")
        self._add_column_if_not_exists(cursor, "users", "last_attempt", "TIMESTAMP")
        self._add_column_if_not_exists(cursor, "users", "last_login", "TIMESTAMP")
        self._add_column_if_not_exists(cursor, "users", "is_locked", "INTEGER DEFAULT 0")
        self._add_column_if_not_exists(cursor, "users", "lock_until", "TIMESTAMP")

    def _add_column_if_not_exists(self, cursor, table_name, column_name, column_type):
        """Utility to add a column to a table if it doesn't exist."""
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            if column_name not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                logger.info(f"Added column '{column_name}' to table '{table_name}'.")
        except sqlite3.Error as e:
            logger.error(f"Failed to add column {column_name} to {table_name}: {e}")

    @contextmanager
    def get_connection(self):
        """Provide a database connection context."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def log_activity(self, username: str, activity: str):
        """Log a user's activity in the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO activity_log (username, activity) VALUES (?, ?)",
                    (username, activity),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log activity for {username}: {e}")


# Singleton instance of the database manager
db_manager = DatabaseManager(DB_PATH)
