"""
Firebase Real-time NPK Sensor Data Calibration Script

FIXES APPLIED:
1. Listener now filters to only UNPROCESSED docs using .where("processed", "!=", True)
   — avoids loading thousands of old documents on startup
2. Removed per-doc baseline logging that caused slowdowns on large collections
3. Added catch-up function to process missed docs on cold start
4. Listener only fires on ADDED changes, ignores MODIFIED/REMOVED
"""

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import threading
import time
import joblib
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_COLLECTION        = os.getenv("RAW_COLLECTION", "npk_readings")
CALIBRATED_COLLECTION = os.getenv("CALIBRATED_COLLECTION", "calibrated_npk_readings_sensor_1")
MODEL_PATH            = os.getenv("MODEL_PATH", "soil_calibration_model.pkl")

# ============================================================================
# INITIALIZE FIREBASE
# ============================================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK using environment variable or local file."""
    try:
        firebase_admin.get_app()
        print("✅ Firebase already initialized")
    except ValueError:
        firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if firebase_creds_json:
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            print("✅ Using Firebase credentials from environment variable")
        else:
            cred = credentials.Certificate(
                "esp32sensor-10e27-firebase-adminsdk-u1xzy-56f4449de5.json"
            )
            print("✅ Using Firebase credentials from local file")

        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully")


initialize_firebase()
db = firestore.client()

# ============================================================================
# LOAD CALIBRATION MODEL
# ============================================================================

def load_model():
    """Load the trained calibration model."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


model = load_model()

# ============================================================================
# DATA PROCESSING
# ============================================================================

def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare sensor data for the model."""
    required = ["N", "P", "K", "pH", "Conductivity"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing field: {key}")

    return pd.DataFrame([[
        data["N"],
        data["P"],
        data["K"],
        data["pH"],
        data["Conductivity"]
    ]], columns=["sensor_N", "sensor_P", "sensor_K", "sensor_PH", "sensor_EC"])


def process_document(doc_id: str, data: dict, raw_collection: str, calibrated_collection: str) -> bool:
    """Calibrate values and upload to Firestore."""
    try:
        print(f"\n🔄 Processing document: {doc_id}")

        # Guard: skip if already calibrated
        calibrated_ref = db.collection(calibrated_collection).document(doc_id)
        if calibrated_ref.get().exists:
            print(f"⏭️  Doc {doc_id} already calibrated, skipping...")
            return False

        # Prepare input & predict
        X = prepare_input(data)
        calibrated = model.predict(X)[0]

        calibrated_dict = {
            "calibrated_N":            float(calibrated[0]),
            "calibrated_P":            float(calibrated[1]),
            "calibrated_K":            float(calibrated[2]),
            "calibrated_pH":           float(calibrated[3]),
            "calibrated_Conductivity": float(calibrated[4]),
            "calibrated_timestamp":    firestore.SERVER_TIMESTAMP,
        }

        # Merge with original data and save
        calibrated_ref.set({**data, **calibrated_dict})

        # Mark original doc as processed
        db.collection(raw_collection).document(doc_id).update({
            "processed":    True,
            "processed_at": firestore.SERVER_TIMESTAMP,
        })

        print(f"✅ Calibrated and saved: {doc_id}")
        print(f"   Collection : {calibrated_collection}")
        print(f"   Sensor ID  : {data.get('sensorId', 'N/A')}")
        print(f"   Timestamp  : {data.get('timestamp', 'N/A')}")
        print(f"   Original   - N:{data['N']}, P:{data['P']}, K:{data['K']}")
        print(
            f"   Calibrated - N:{calibrated_dict['calibrated_N']:.2f}, "
            f"P:{calibrated_dict['calibrated_P']:.2f}, "
            f"K:{calibrated_dict['calibrated_K']:.2f}"
        )
        return True

    except KeyError as e:
        print(f"❌ Missing field in {doc_id}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# CATCH-UP: process missed docs from cold start / downtime
# ============================================================================

def process_missed_documents(raw_collection: str, calibrated_collection: str):
    """
    On startup, find and process any documents that arrived while
    the service was down (cold start gap).
    Looks for docs where processed != True.
    """
    print(f"\n🔍 [{raw_collection}] Checking for missed documents...")

    try:
        docs = list(
            db.collection(raw_collection)
            .where("processed", "!=", True)
            .stream()
        )

        if not docs:
            print(f"✅ [{raw_collection}] No missed documents found")
            return

        print(f"⚠️  [{raw_collection}] Found {len(docs)} missed document(s) — processing now...")

        success_count = 0
        for doc in docs:
            success = process_document(
                doc.id, doc.to_dict(), raw_collection, calibrated_collection
            )
            if success:
                success_count += 1

        print(f"✅ [{raw_collection}] Catch-up complete: {success_count}/{len(docs)} processed")

    except Exception as e:
        print(f"❌ [{raw_collection}] Catch-up error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# REAL-TIME LISTENER
# ============================================================================

def start_listener_for_collections(raw_collection: str, calibrated_collection: str, stop_event: threading.Event):
    """
    Start a Firestore listener for a specific collection pair.

    KEY FIXES:
    1. Queries only unprocessed docs (.where processed != True)
       so initial snapshot is small, not the entire collection history
    2. Only processes ADDED change types — ignores MODIFIED (processed=True update)
       and REMOVED to prevent infinite loops
    3. Catch-up runs first to handle cold-start missed documents
    """

    # ── Step 1: Process anything missed while service was down ──────────
    process_missed_documents(raw_collection, calibrated_collection)

    # ── Step 2: Start live listener for new documents ────────────────────
    initial_snapshot_received = False

    def on_snapshot(doc_snapshot, changes, read_time):
        nonlocal initial_snapshot_received

        # Skip initial snapshot — catch-up already handled missed docs
        if not initial_snapshot_received:
            initial_snapshot_received = True
            print(f"📦 [{raw_collection}] Listener active — watching for new documents")
            return

        # Only process genuinely NEW documents
        for change in changes:
            if change.type.name != "ADDED":
                # MODIFIED  → fires when we set processed=True, ignore
                # REMOVED   → fires on deletes, ignore
                continue

            doc  = change.document
            data = doc.to_dict()

            print(f"\n📨 [{raw_collection}] New document detected: {doc.id}")
            process_document(doc.id, data, raw_collection, calibrated_collection)

    print(f"👀 [{raw_collection}] Starting live listener → {calibrated_collection}")

    # FIX: Only listen to unprocessed docs — avoids loading full collection history
    doc_watch = (
        db.collection(raw_collection)
        .where("processed", "!=", True)
        .on_snapshot(on_snapshot)
    )

    # Keep thread alive
    while not stop_event.is_set():
        time.sleep(1)

    # Clean shutdown
    if doc_watch:
        doc_watch.unsubscribe()
        print(f"🛑 Stopped listener for {raw_collection}")

# ============================================================================
# WORKER THREAD MANAGEMENT
# ============================================================================

worker_threads: list[threading.Thread] = []
stop_event     = threading.Event()
worker_running = False


def start_worker() -> str:
    """Start background listeners for all sensor collections."""
    global worker_threads, stop_event, worker_running

    if worker_running:
        print("⚠️  Worker already running")
        return "Worker already running"

    stop_event.clear()
    worker_running = True

    print("=" * 70)
    print("🚀 NPK Calibration Service Started")
    print("=" * 70)
    print(f"🤖 Model     : {MODEL_PATH}")
    print(f"🌍 Environment: {'PRODUCTION (Render)' if os.getenv('RENDER') else 'LOCAL'}")
    print("=" * 70)

    # Sensor 1
    thread1 = threading.Thread(
        target=start_listener_for_collections,
        args=("npk_readings", "calibrated_npk_readings_sensor_1", stop_event),
        daemon=True,
        name="listener-sensor-1",
    )
    thread1.start()
    worker_threads.append(thread1)

    # Sensor 2
    thread2 = threading.Thread(
        target=start_listener_for_collections,
        args=("npk_readings_sensor_2", "calibrated_npk_readings_sensor_2", stop_event),
        daemon=True,
        name="listener-sensor-2",
    )
    thread2.start()
    worker_threads.append(thread2)

    print("\n✅ All listeners started successfully")
    print("✅ Monitoring: npk_readings + npk_readings_sensor_2")
    print("=" * 70)
    return "Worker started successfully"


def stop_worker() -> str:
    """Stop all background workers gracefully."""
    global worker_threads, stop_event, worker_running

    if not worker_running:
        print("⚠️  Worker not running")
        return "Worker not running"

    print("\n🛑 Stopping workers...")
    stop_event.set()

    for thread in worker_threads:
        thread.join(timeout=5)

    worker_threads.clear()
    worker_running = False
    print("✅ All workers stopped")
    return "Worker stopped successfully"

# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    start_worker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Keyboard interrupt received...")
        stop_worker()
        print("✅ Service stopped. Exiting...")
