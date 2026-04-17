"""
Streamlit authentication interface
"""

import streamlit as st
from auth_service import login_user, register_user


def authentication_ui():

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    menu = st.sidebar.selectbox(
        "Account",
        ["Login", "Register"]
    )

    # ============================
    # REGISTER
    # ============================

    if menu == "Register":

        st.subheader("Create Account")

        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Register"):

            if register_user(username, email, password):

                st.success("Account created successfully")

            else:
                st.error("Username or email already exists")

    # ============================
    # LOGIN
    # ============================

    if menu == "Login":

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            user_id = login_user(username, password)

            if user_id:

                st.session_state.user_id = user_id
                st.success("Login successful")

                st.rerun()

            else:
                st.error("Invalid username or password")

    return st.session_state.user_id