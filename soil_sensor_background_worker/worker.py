"""
Firebase Real-time NPK Sensor Data Calibration Script
Listens to new documents in npk_readings collection and processes them
through the calibration model, then stores results in calibrated_npk_readings
"""

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import datetime
import threading
import time
import joblib
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Firebase collections
RAW_COLLECTION = os.getenv("RAW_COLLECTION", "npk_readings")
CALIBRATED_COLLECTION = os.getenv("CALIBRATED_COLLECTION", "calibrated_npk_readings_sensor_1")

# Path to your trained model file
MODEL_PATH = "soil_calibration_model.pkl"

# ============================================================================
# INITIALIZE FIREBASE
# ============================================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK using environment variable"""
    try:
        firebase_admin.get_app()
        print("✅ Firebase already initialized")
    except ValueError:
        # Try to get credentials from environment variable (for Render)
        firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        
        if firebase_creds_json:
            # Parse JSON string from environment variable
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            print("✅ Using Firebase credentials from environment variable")
        else:
            # Fall back to local file (for local development)
            cred = credentials.Certificate("esp32sensor-10e27-firebase-adminsdk-u1xzy-56f4449de5.json")
            print("✅ Using Firebase credentials from local file")
        
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully")

initialize_firebase()
db = firestore.client()

# Track the last processed document ID to avoid duplicates
last_processed_doc_id = None
processing_lock = threading.Lock()
initial_snapshot_received = False

# ============================================================================
# LOAD CALIBRATION MODEL
# ============================================================================

def load_model():
    """Load the trained calibration model"""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

model = load_model()

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare sensor data for the model"""
    required = ["N", "P", "K", "pH", "Conductivity"]
    
    for key in required:
        if key not in data:
            raise KeyError(f"Missing field: {key}")

    df = pd.DataFrame([[
        data["N"],
        data["P"],
        data["K"],
        data["pH"],
        data["Conductivity"]
    ]], columns=["sensor_N", "sensor_P", "sensor_K", "sensor_PH", "sensor_EC"])
    
    return df


def process_document(doc_id: str, data: dict):
    """Calibrate values and upload to Firestore"""
    try:
        print(f"\n🔄 Processing document: {doc_id}")
        
        # Check if already calibrated in database
        calibrated_ref = db.collection(CALIBRATED_COLLECTION).document(doc_id)
        if calibrated_ref.get().exists:
            print(f"⏭️  Doc {doc_id} already exists in calibrated collection, skipping...")
            return False

        # Check if marked as processed
        if data.get('processed') == True:
            print(f"⏭️  Doc {doc_id} already marked as processed, skipping...")
            return False

        # Prepare input & predict
        X = prepare_input(data)
        calibrated = model.predict(X)[0]

        # Create calibrated data dictionary
        calibrated_dict = {
            "calibrated_N": float(calibrated[0]),
            "calibrated_P": float(calibrated[1]),
            "calibrated_K": float(calibrated[2]),
            "calibrated_pH": float(calibrated[3]),
            "calibrated_Conductivity": float(calibrated[4]),
            "calibrated_timestamp": firestore.SERVER_TIMESTAMP
        }

        # Merge with original data
        merged = {**data, **calibrated_dict}

        # Save to calibrated_npk_readings/{same_doc_id}
        calibrated_ref.set(merged)
        
        # Mark the original document as processed
        db.collection(RAW_COLLECTION).document(doc_id).update({
            'processed': True,
            'processed_at': firestore.SERVER_TIMESTAMP
        })
        
        print(f"✅ Calibrated and saved doc {doc_id}")
        print(f"   Sensor ID: {data.get('sensorId', 'N/A')}")
        print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"   Original - N:{data['N']}, P:{data['P']}, K:{data['K']}")
        print(f"   Calibrated - N:{calibrated_dict['calibrated_N']:.2f}, "
              f"P:{calibrated_dict['calibrated_P']:.2f}, "
              f"K:{calibrated_dict['calibrated_K']:.2f}")
        
        return True

    except KeyError as e:
        print(f"❌ Missing field in document {doc_id}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_latest_document_by_timestamp():
    """
    Query the collection for the latest document by timestamp (descending order)
    and return only the first one
    """
    try:
        # Query: Get documents ordered by timestamp descending, limit to 1
        docs = (db.collection(RAW_COLLECTION)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream())
        
        latest_doc = None
        for doc in docs:
            latest_doc = doc
            break
        
        if latest_doc:
            return latest_doc
        else:
            return None
    
    except Exception as e:
        print(f"❌ Error fetching latest document: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# REAL-TIME LISTENER
# ============================================================================

def on_snapshot(doc_snapshot, changes, read_time):
    """
    Callback when ANY change happens in the collection.
    Skips initial snapshot, then queries for latest document on real changes.
    """
    global last_processed_doc_id, initial_snapshot_received
    
    # Skip the initial snapshot (all existing documents)
    if not initial_snapshot_received:
        initial_snapshot_received = True
        print(f"📦 Initial snapshot received ({len(changes)} existing documents)")
        print(f"✅ Skipping initial snapshot - listener now active for NEW changes\n")
        return
    
    # Use lock to prevent concurrent processing
    with processing_lock:
        print(f"\n🔔 Change detected in collection at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Number of changes: {len(changes)}")
        
        # Log what changed (only first 3 to avoid spam)
        for i, change in enumerate(changes[:3]):
            change_type = change.type.name
            doc_id = change.document.id
            print(f"   - {change_type}: {doc_id}")
        
        if len(changes) > 3:
            print(f"   ... and {len(changes) - 3} more changes")
        
        # Now query for the LATEST document by timestamp (only one)
        print(f"\n🔍 Querying for latest document by timestamp (DESC, LIMIT 1)...")
        latest_doc = get_latest_document_by_timestamp()
        
        if latest_doc:
            doc_id = latest_doc.id
            data = latest_doc.to_dict()
            
            print(f"📊 Latest document: {doc_id}")
            print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"   Sensor ID: {data.get('sensorId', 'N/A')}")
            
            # Check if this is a different document than the last one we processed
            if doc_id != last_processed_doc_id:
                print(f"\n✨ New latest document detected!")
                
                # Process the latest document
                success = process_document(doc_id, data)
                
                if success:
                    # Update last processed ID
                    last_processed_doc_id = doc_id
            else:
                print(f"⏭️  Latest document was already processed (ID: {doc_id})")
        
        print(f"\n👀 Waiting for next change...")


def get_current_latest_document():
    """Get the current latest document to set as baseline"""
    try:
        docs = (db.collection(RAW_COLLECTION)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream())
        
        for doc in docs:
            return doc
        return None
    except Exception as e:
        print(f"❌ Error fetching baseline document: {e}")
        return None


def start_listener():
    """Start real-time listener for the collection"""
    global last_processed_doc_id
    
    print("="*70)
    print("🚀 NPK Calibration Service Started")
    print("="*70)
    print(f"📡 Monitoring collection: {RAW_COLLECTION}")
    print(f"💾 Saving calibrated data to: {CALIBRATED_COLLECTION}")
    print(f"🤖 Model loaded and ready")
    print(f"🌍 Environment: {'PRODUCTION (Render)' if os.getenv('RENDER') else 'LOCAL'}")
    print("="*70)
    print("\n🔍 Getting current latest document as baseline...")
    
    # Get the current latest document to set as baseline
    baseline_doc = get_current_latest_document()
    if baseline_doc:
        last_processed_doc_id = baseline_doc.id
        data = baseline_doc.to_dict()
        print(f"✅ Baseline set: {baseline_doc.id}")
        print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"   Sensor ID: {data.get('sensorId', 'N/A')}")
        print(f"   (This document will NOT be processed unless new data arrives)")
    else:
        print("ℹ️  No existing documents found. Will process first incoming document.")
    
    print("\n✅ Starting real-time listener...")
    print("✅ Skips initial snapshot (existing documents)")
    print("✅ Queries for latest document by timestamp on each change")
    print("✅ Processes ONLY the most recent document")
    print("✅ Duplicate prevention enabled")
    print("="*70)
    print("\n⏳ Attaching listener...\n")
    
    # Attach listener to the ENTIRE collection
    collection_ref = db.collection(RAW_COLLECTION)
    doc_watch = collection_ref.on_snapshot(on_snapshot)
    
    print("👀 Listener active - waiting for NEW database changes...\n")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping listener...")
        doc_watch.unsubscribe()
        print("✅ Listener stopped. Exiting...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
worker_thread = None
running = False

def start_worker():
    """Start background Firebase listener in a thread"""
    global worker_thread, running
    if running:
        print("⚠️ Worker already running")
        return "Worker already running"
    
    running = True
    worker_thread = threading.Thread(target=start_listener, daemon=True)
    worker_thread.start()
    print("🚀 Worker started in background thread")
    return "Worker started"

def stop_worker():
    """Stop the background worker gracefully"""
    global running
    running = False
    print("🛑 Worker stop requested (will stop after current loop)")
    return "Worker stopping"
