# firebase_utils.py
import firebase_admin
from firebase_admin import credentials, firestore
import time

def init_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

def update_firestore(angle, dropcount, success):
    user_id = "user_123"
    data = {
        "angle": angle if angle is not None else 0,
        "drops_left_eye": dropcount,
        "success": success,
        "timestamp": int(time.time())
    }
    try:
        doc_ref = db.collection("Users").document(user_id).collection("administration_records").document()
        doc_ref.set(data)
        print("✅ Firestore updated with:", data)
        return doc_ref.id
    except Exception as e:
        print("❌ Error updating Firestore:", e)
        return None
