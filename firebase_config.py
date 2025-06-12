import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

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

# Firebase Admin SDK for Firestore (corrected file name)
cred = credentials.Certificate("firebase-service-account.json")  # ✅ UPDATE this line
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
