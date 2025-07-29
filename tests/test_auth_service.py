"""Unit tests for AuthService."""

import pytest
from src.services.auth_service import AuthService


@pytest.fixture
def auth_service():
    return AuthService()


def test_register_and_login(auth_service):
    username = "testuser"
    password = "testpass123"
    # Register user
    assert auth_service.register(username, password) is True
    # Login with correct credentials
    assert auth_service.login(username, password) is True
    # Login with wrong password
    assert auth_service.login(username, "wrongpass") is False
