from fastapi import FastAPI
from worker import start_worker, stop_worker

app = FastAPI(title="NPK Calibration FastAPI Server")

@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI server started. Launching worker...")
    start_worker()

@app.on_event("shutdown")
def shutdown_event():
    print("🛑 FastAPI shutting down. Stopping worker...")
    stop_worker()

@app.get("/")
def home():
    return {"status": "running", "message": "NPK Calibration Service is active"}

@app.post("/start")
def start():
    msg = start_worker()
    return {"status": msg}

@app.post("/stop")
def stop():
    msg = stop_worker()
    return {"status": msg}

@app.get("/health")
def health():
    return {"status": "ok", "message": "Server is healthy"}
