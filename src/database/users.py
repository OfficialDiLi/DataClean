"""User database operations."""

from datetime import datetime, timedelta
import bcrypt
import uuid
from typing import Optional, Tuple
from src.database.database import db_manager
from src.utils.logging_utils import logger
from src.config.config import MAX_LOGIN_ATTEMPTS, LOGIN_COOLDOWN


class UserDB:
    @staticmethod
    def create_user(username: str, password: str) -> bool:
        """Create a new user account."""
        try:
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed_password),
                )
                conn.commit()

            db_manager.log_activity(username, "account_created")
            return True

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False

    @staticmethod
    def validate_login(username: str, password: str) -> Tuple[bool, Optional[str]]:
        """Validate user login credentials and handle login attempts."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT password, login_attempts, lock_until, is_locked 
                    FROM users 
                    WHERE username = ?
                    """,
                    (username,),
                )
                result = cursor.fetchone()

                if not result:
                    return False, "Invalid credentials"

                stored_password, attempts, lock_until, is_locked = result

                # Check if account is locked
                if is_locked:
                    if (
                        lock_until
                        and datetime.fromisoformat(lock_until) > datetime.now()
                    ):
                        return False, "Account is locked. Try again later."
                    else:
                        # Reset lock if cooldown period has passed
                        cursor.execute(
                            """
                            UPDATE users 
                            SET is_locked = 0, login_attempts = 0, lock_until = NULL 
                            WHERE username = ?
                            """,
                            (username,),
                        )
                        conn.commit()

                # Verify password
                if bcrypt.checkpw(password.encode("utf-8"), stored_password):
                    # Reset login attempts on successful login
                    cursor.execute(
                        """
                        UPDATE users 
                        SET login_attempts = 0, 
                            last_login = CURRENT_TIMESTAMP,
                            is_locked = 0,
                            lock_until = NULL
                        WHERE username = ?
                        """,
                        (username,),
                    )
                    conn.commit()

                    db_manager.log_activity(username, "login_success")
                    return True, None

                # Handle failed login attempt
                new_attempts = attempts + 1
                if new_attempts >= MAX_LOGIN_ATTEMPTS:
                    lock_until = datetime.now() + timedelta(seconds=LOGIN_COOLDOWN)
                    cursor.execute(
                        """
                        UPDATE users 
                        SET login_attempts = ?, 
                            is_locked = 1,
                            lock_until = ?,
                            last_attempt = CURRENT_TIMESTAMP
                        WHERE username = ?
                        """,
                        (new_attempts, lock_until, username),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE users 
                        SET login_attempts = ?,
                            last_attempt = CURRENT_TIMESTAMP
                        WHERE username = ?
                        """,
                        (new_attempts, username),
                    )
                conn.commit()

                db_manager.log_activity(username, "login_failed")
                return False, "Invalid credentials"

        except Exception as e:
            logger.error(f"Login validation error: {e}")
            return False, "An error occurred during login"

    @staticmethod
    def create_session(username: str) -> Optional[str]:
        """Create a new session for the user."""
        try:
            session_id = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(days=1)

            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_sessions (session_id, username, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, username, expires_at),
                )
                conn.commit()

            return session_id

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None

    @staticmethod
    def validate_session(session_id: str) -> bool:
        """Validate a user session."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 1 FROM user_sessions 
                    WHERE session_id = ? 
                    AND expires_at > datetime('now')
                    AND is_active = 1
                    """,
                    (session_id,),
                )
                return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False
