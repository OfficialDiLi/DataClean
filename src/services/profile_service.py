"""User profile management service."""

from typing import Optional
from src.database.users import UserDB


class ProfileService:
    @staticmethod
    def get_profile(username: str) -> Optional[dict]:
        # Dummy implementation, replace with actual DB query
        return {"username": username, "bio": "", "email": ""}

    @staticmethod
    def update_profile(username: str, bio: str, email: str) -> bool:
        # Dummy implementation, replace with actual DB update
        return True
