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


def process_document(doc_id: str, data: dict, 
                     RAW_COLLECTION, 
                     CALIBRATED_COLLECTION):
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
        print(f"   Collection: {CALIBRATED_COLLECTION}")
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


# ============================================================================
# REAL-TIME LISTENER
# ============================================================================

def start_listener_for_collections(RAW_COLLECTION, CALIBRATED_COLLECTION, stop_event):
    """Start listener for a specific collection pair"""
    last_processed_doc_id = None
    initial_snapshot_received = False
    processing_lock = threading.Lock()
    doc_watch = None

    def get_latest_document():
        """Get the most recent document from the raw collection"""
        docs = (db.collection(RAW_COLLECTION)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream())
        for doc in docs:
            return doc
        return None

    def on_snapshot(doc_snapshot, changes, read_time):
        """Callback for Firestore snapshot listener"""
        nonlocal last_processed_doc_id, initial_snapshot_received

        # Skip the initial snapshot to avoid processing old data
        if not initial_snapshot_received:
            initial_snapshot_received = True
            print(f"📦 [{RAW_COLLECTION}] Initial snapshot received and skipped")
            
            # Set baseline to current latest document
            baseline_doc = get_latest_document()
            if baseline_doc:
                last_processed_doc_id = baseline_doc.id
                data = baseline_doc.to_dict()
                print(f"✅ Baseline set: {baseline_doc.id}")
                print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
                print(f"   Sensor ID: {data.get('sensorId', 'N/A')}")
            return

        # Process only the latest document
        with processing_lock:
            latest_doc = get_latest_document()
            if not latest_doc:
                return

            # Skip if already processed
            if latest_doc.id == last_processed_doc_id:
                return

            # Process the document
            success = process_document(
                latest_doc.id,
                latest_doc.to_dict(),
                RAW_COLLECTION,
                CALIBRATED_COLLECTION
            )

            # Update last processed ID on success
            if success:
                last_processed_doc_id = latest_doc.id

    print(f"👀 Starting listener for {RAW_COLLECTION} → {CALIBRATED_COLLECTION}")
    doc_watch = db.collection(RAW_COLLECTION).on_snapshot(on_snapshot)
    
    # Keep the listener alive until stop_event is set
    while not stop_event.is_set():
        time.sleep(1)
    
    # Unsubscribe when stopping
    if doc_watch:
        doc_watch.unsubscribe()
        print(f"🛑 Stopped listener for {RAW_COLLECTION}")


# ============================================================================
# WORKER THREAD MANAGEMENT
# ============================================================================

# Global state for worker threads
worker_threads = []
stop_event = threading.Event()
worker_running = False

def start_worker():
    """Start background workers for all sensor collections"""
    global worker_threads, stop_event, worker_running
    
    if worker_running:
        print("⚠️  Worker already running")
        return "Worker already running"
    
    # Reset stop event
    stop_event.clear()
    worker_running = True
    
    print("="*70)
    print("🚀 NPK Calibration Service Started")
    print("="*70)
    print(f"🤖 Model loaded and ready")
    print(f"🌍 Environment: {'PRODUCTION (Render)' if os.getenv('RENDER') else 'LOCAL'}")
    print("="*70)
    
    # Start listener for Sensor 1
    thread1 = threading.Thread(
        target=start_listener_for_collections,
        args=("npk_readings", "calibrated_npk_readings_sensor_1", stop_event),
        daemon=True
    )
    thread1.start()
    worker_threads.append(thread1)

    # Start listener for Sensor 2
    thread2 = threading.Thread(
        target=start_listener_for_collections,
        args=("npk_readings_sensor_2", "calibrated_npk_readings_sensor_2", stop_event),
        daemon=True
    )
    thread2.start()
    worker_threads.append(thread2)

    print("\n✅ All listeners started successfully")
    print("✅ Monitoring both sensor collections")
    print("="*70)
    
    return "Worker started successfully"


def stop_worker():
    """Stop the background worker gracefully"""
    global worker_threads, stop_event, worker_running
    
    if not worker_running:
        print("⚠️  Worker not running")
        return "Worker not running"
    
    print("\n🛑 Stopping workers...")
    stop_event.set()
    
    # Wait for all threads to finish (with timeout)
    for thread in worker_threads:
        thread.join(timeout=5)
    
    worker_threads.clear()
    worker_running = False
    
    print("✅ All workers stopped")
    return "Worker stopped successfully"


# ============================================================================
# STANDALONE ENTRY POINT (for direct execution)
# ============================================================================

if __name__ == "__main__":
    start_worker()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Keyboard interrupt received...")
        stop_worker()
        print("✅ Service stopped. Exiting...")