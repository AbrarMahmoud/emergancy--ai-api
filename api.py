'''from fastapi import FastAPI, UploadFile, File
import shutil
import uuid

from ai_model import (
    detect_fire,
    detect_accident,
    classify_sos_text
)
from alert_engine import fire_alert, accident_alert, sos_alert

app = FastAPI(title="Emergency AI Service")

# ================= FIRE =================
@app.post("/predict/fire")
async def fire_detection(file: UploadFile = File(...)):
    temp_name = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    count = detect_fire(temp_name)
    alert = fire_alert(count)

    return {
        "type": "fire",
        "fire_count": count,
        "alert": alert
    }


# ================= ACCIDENT =================
@app.post("/predict/accident")
async def accident_detection(file: UploadFile = File(...)):
    temp_name = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    count = detect_accident(temp_name)
    alert = accident_alert(count)

    return {
        "type": "accident",
        "accident_count": count,
        "alert": alert
    }


# ================= SOS TEXT =================
@app.post("/predict/sos")
async def sos_detection(text: str):
    label, confidence = classify_sos_text(text)
    alert = sos_alert(label)

    return {
        "type": "sos",
        "label": label,
        "confidence": confidence,
        "alert": alert
    }'''
    
    
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
import uuid
import cv2
import os
from pathlib import Path

from ai_model import detect_fire, detect_accident, classify_sos_text
from alert_engine import fire_alert, accident_alert, sos_alert

app = FastAPI(title="Emergency AI Service")

# ================= Helper Function =================
def process_video(file_path: str, detect_function):
    cap = cv2.VideoCapture(file_path)
    total_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_frame = f"frame_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_frame, frame)
        total_count += detect_function(temp_frame)
        os.remove(temp_frame)
    cap.release()
    return total_count

# ================= Fire Detection (Image or Video) =================
@app.post("/predict/fire")
async def fire_detection(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".mp4", ".avi"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_file = f"temp_{uuid.uuid4()}{ext}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext in [".mp4", ".avi"]:
        count = process_video(temp_file, detect_fire)
    else:
        count = detect_fire(temp_file)

    os.remove(temp_file)
    alert = fire_alert(count)

    return {
        "type": "fire",
        "fire_count": count,
        "alert": alert
    }


# ================= Accident Detection (Image or Video) =================
@app.post("/predict/accident")
async def accident_detection(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".mp4", ".avi"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_file = f"temp_{uuid.uuid4()}{ext}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext in [".mp4", ".avi"]:
        count = process_video(temp_file, detect_accident)
    else:
        count = detect_accident(temp_file)

    os.remove(temp_file)
    alert = accident_alert(count)

    return {
        "type": "accident",
        "accident_count": count,
        "alert": alert
    }


# ================= SOS Detection (Text) =================
@app.post("/predict/sos")
async def sos_detection(text: str = Form(...)):
    label, confidence = classify_sos_text(text)
    alert = sos_alert(label)

    return {
        "type": "sos",
        "label": label,
        "confidence": confidence,
        "alert": alert
    }