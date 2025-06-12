import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

# Pyrebase config (for login/signup)
firebaseConfig = {
    "apiKey": "AIzaSyA6vCseYRF8bw0NgU9V3yVUYOXDnOBrIjM",
    "authDomain": "movieai-authentication.firebaseapp.com",
    "projectId": "movieai-authentication",
    "storageBucket": "movieai-authentication.appspot.com",
    "messagingSenderId": "61929349840",
    "appId": "1:61929349840:web:55b47dd4148d7b3dd2f4d5",
    "measurementId": "G-GSMZFQW9JJ",
    "databaseURL": ""  # leave this empty
}

# Pyrebase for auth
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Firebase Admin SDK using Streamlit secrets
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

cred = credentials.Certificate(firebase_secrets)
firebase_admin.initialize_app(cred)

# Firestore DB client
db = firestore.client()

# Save data to Firestore
def save_user_data(email, fav_movies, recs):
    doc_ref = db.collection("users").document(email)
    doc_ref.set({
        "favorites": fav_movies,
        "recommendations": recs
    })

# Load data from Firestore
def load_user_data(email):
    doc_ref = db.collection("users").document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None
