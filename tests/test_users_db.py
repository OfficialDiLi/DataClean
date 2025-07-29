"""Unit tests for UserDB."""

import pytest
from src.database.users import UserDB


def test_create_user():
    assert UserDB.create_user("pytestuser", "pytestpass") is True


def test_validate_login():
    UserDB.create_user("pytestlogin", "pytestpass")
    success, msg = UserDB.validate_login("pytestlogin", "pytestpass")
    assert success is True
    fail, msg = UserDB.validate_login("pytestlogin", "wrongpass")
    assert fail is False
