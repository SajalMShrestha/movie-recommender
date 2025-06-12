import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
import streamlit as st
import json
import os
import tempfile

# Convert secrets to a temporary JSON file
firebase_dict = {
    "type": st.secrets["firebase_type"],
    "project_id": st.secrets["firebase_project_id"],
    "private_key_id": st.secrets["firebase_private_key_id"],
    "private_key": st.secrets["firebase_private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["firebase_client_email"],
    "client_id": st.secrets["firebase_client_id"],
    "auth_uri": st.secrets["firebase_auth_uri"],
    "token_uri": st.secrets["firebase_token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase_auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase_client_x509_cert_url"]
}

# Write to a temp file
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
    json.dump(firebase_dict, f)
    temp_filename = f.name

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(temp_filename)
    firebase_admin.initialize_app(cred)

# Auth and Firestore
auth = firebase_auth
db = firestore.client()

def save_user_data(email, favorites, recommendations):
    doc_ref = db.collection("users").document(email)
    doc_ref.set({
        "favorites": favorites,
        "recommendations": recommendations
    })

def load_user_data(email):
    doc_ref = db.collection("users").document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None
