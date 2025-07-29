"""Password reset service."""

from src.database.users import UserDB


class PasswordService:
    @staticmethod
    def reset_password(username: str, new_password: str) -> bool:
        # Dummy implementation, replace with actual DB update
        return True
