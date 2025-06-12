import streamlit as st
from firebase_config import auth

def login_signup_ui():
    st.title("🔐 Welcome to Movie AI Recommender")

    tab1, tab2 = st.tabs(["Log In", "Sign Up"])

    with tab1:
        st.subheader("Log In")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In"):
            try:
                user = auth.sign_in_with_email_and_password(login_email, login_password)
                st.session_state["user"] = user
                st.session_state["email"] = login_email
                st.success("Logged in successfully.")
                st.experimental_rerun()
            except:
                st.error("Login failed. Check your credentials.")

    with tab2:
        st.subheader("Sign Up")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            try:
                user = auth.create_user_with_email_and_password(signup_email, signup_password)
                st.success("Account created! Please log in.")
            except:
                st.error("Sign-up failed. Try again with a different email or stronger password.")
