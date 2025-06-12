import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
import streamlit as st

# Load Firebase secrets from Streamlit secrets
firebase_secrets = {
    "type": st.secrets["firebase_type"],
    "project_id": st.secrets["firebase_project_id"],
    "private_key_id": st.secrets["firebase_private_key_id"],
    "private_key": st.secrets["firebase_private_key"],
    "client_email": st.secrets["firebase_client_email"],
    "client_id": st.secrets["firebase_client_id"],
    "auth_uri": st.secrets["firebase_auth_uri"],
    "token_uri": st.secrets["firebase_token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase_auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase_client_x509_cert_url"]
}

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

# Firebase auth
auth = firebase_auth
db = firestore.client()

# Save user data to Firestore
def save_user_data(email, favorites, recommendations):
    doc_ref = db.collection("users").document(email)
    doc_ref.set({
        "favorites": favorites,
        "recommendations": recommendations
    })

# Load user data from Firestore
def load_user_data(email):
    doc_ref = db.collection("users").document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None
