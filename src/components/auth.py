"""Authentication related components."""

from typing import Optional, Tuple
import streamlit as st
from src.services.auth_service import AuthService


def show_login_form() -> Tuple[str, str]:
    """Display login form and return credentials."""
    username = st.text_input("User Name")
    password = st.text_input("Password", type="password")
    return username, password


def show_signup_form() -> Tuple[str, str]:
    """Display signup form and return credentials."""
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    return new_user, new_password


def auth_page():
    """Main authentication page component."""
    st.title("Welcome to DataClean Pro")
    menu = ["Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    auth_service = AuthService()

    if choice == "Login":
        st.subheader("Login Section")
        username, password = show_login_form()

        if st.button("Login"):
            if auth_service.login(username, password):
                st.session_state.logged_in = True
                st.success(f"Welcome {username}")
                st.rerun()
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user, new_password = show_signup_form()

        if st.button("Signup"):
            if auth_service.register(new_user, new_password):
                st.success("You have successfully created an account")
                st.info("Go to Login Menu to login")
            else:
                st.warning("Username already exists")
