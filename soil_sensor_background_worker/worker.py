"""
Firebase Real-time NPK Sensor Data Calibration Script

Listens to new documents in npk_readings collection and processes them
through the calibration model, then stores results in calibrated_npk_readings

FIX APPLIED:
- Listener now uses `changes` list with `change.type.name == "ADDED"` filter
  instead of fetching the "latest" document on every snapshot trigger.
  This prevents re-processing when old docs are updated (e.g. processed=True flag).
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
# DATA PROCESSING FUNCTIONS
# ============================================================================

def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare sensor data for the model."""
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


def process_document(doc_id: str, data: dict, raw_collection: str, calibrated_collection: str) -> bool:
    """Calibrate values and upload to Firestore."""
    try:
        print(f"\n🔄 Processing document: {doc_id}")

        # Guard: skip if already calibrated
        calibrated_ref = db.collection(calibrated_collection).document(doc_id)
        if calibrated_ref.get().exists:
            print(f"⏭️  Doc {doc_id} already exists in calibrated collection, skipping...")
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
        merged = {**data, **calibrated_dict}
        calibrated_ref.set(merged)

        # Mark the original document as processed
        db.collection(raw_collection).document(doc_id).update({
            "processed":    True,
            "processed_at": firestore.SERVER_TIMESTAMP,
        })

        print(f"✅ Calibrated and saved doc {doc_id}")
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
        print(f"❌ Missing field in document {doc_id}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# REAL-TIME LISTENER  (THE CRITICAL FIX IS HERE)
# ============================================================================

def start_listener_for_collections(raw_collection: str, calibrated_collection: str, stop_event: threading.Event):
    """
    Start a Firestore listener for a specific collection pair.

    KEY FIX:
      Previously, on_snapshot fetched the "latest" document on every trigger,
      meaning updates to old documents (e.g. setting processed=True) would
      re-fire the callback and risk infinite loops / duplicate processing.

      Now we iterate over `changes` and only process documents whose
      change type is "ADDED" — i.e. genuinely new documents written to the
      collection. MODIFIED and REMOVED changes are ignored entirely.
    """

    initial_snapshot_received = False
    doc_watch = None

    def on_snapshot(doc_snapshot, changes, read_time):
        nonlocal initial_snapshot_received

        # ── Skip the very first snapshot (contains existing docs, not new ones) ──
        if not initial_snapshot_received:
            initial_snapshot_received = True
            print(f"📦 [{raw_collection}] Initial snapshot received — skipping existing docs")

            # Log baseline info for transparency
            for doc in doc_snapshot:
                data = doc.to_dict()
                print(f"   Baseline doc: {doc.id} | "
                      f"Timestamp: {data.get('timestamp', 'N/A')} | "
                      f"Sensor: {data.get('sensorId', 'N/A')}")
            return

        # ── THE FIX: only act on genuinely NEW documents ──────────────────────
        for change in changes:
            # change.type is a DocumentChange enum: ADDED, MODIFIED, REMOVED
            if change.type.name != "ADDED":
                # MODIFIED fires when we update processed=True — ignore it
                # REMOVED fires on deletes — ignore it
                continue

            doc  = change.document
            data = doc.to_dict()

            print(f"\n📨 [{raw_collection}] New document detected: {doc.id}")
            process_document(doc.id, data, raw_collection, calibrated_collection)

    print(f"👀 Starting listener: {raw_collection} → {calibrated_collection}")
    doc_watch = db.collection(raw_collection).on_snapshot(on_snapshot)

    # Keep the thread alive until the stop_event is set
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
stop_event    = threading.Event()
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
    print(f"🤖 Model: {MODEL_PATH}")
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
    print("✅ Monitoring both sensor collections")
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