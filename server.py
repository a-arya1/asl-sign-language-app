

"""
server.py - FastAPI backend for ASL prediction with DTW for J/Z
Run with: uvicorn server:app --reload --port 8000
"""
 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from normalize_data import normalize_landmarks, get_angle_features
from collections import deque
import time
from dtaidistance import dtw
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import uuid
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
import os

MODEL_PATH = "hand_gesture_model.joblib"
CONTRIBUTIONS_DIR = Path("contributions/pending")
CONTRIBUTE_CONF_THRESHOLD = 0.60
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    import gdown
    gdown.download(id="1yg_r-LzS1uEODFdtGmgB0snOu5-Wka64", output=MODEL_PATH, quiet=False)
    print("Download complete.")
model = joblib.load(MODEL_PATH)
print(f"Model loaded. Classes: {list(model.classes_)}")
 
# Load J/Z templates
def load_templates(letter, count=10):
    templates = []
    for i in range(count):
        try:
            t = np.load(f"templates/{letter}/template_{i}.npy")
            templates.append(t)
        except FileNotFoundError:
            pass
    print(f"Loaded {len(templates)} templates for {letter}")
    return templates
 
jtemplates = load_templates("J")
ztemplates = load_templates("Z")
 
# Per-client state (single user assumed for local use)
wrist_buffer = deque(maxlen=30)
dtw_cooldown = 0
DTW_COOLDOWN_SECS = 2.0
DTW_THRESH = 0.15
 
 
class LandmarkData(BaseModel):
    landmarks: list       # flat [x,y,z ...] 63 values
    landmarks_2d: list    # [[x,y] ...] 21 pairs
 

def build_prediction(landmarks, landmarks_2d, track_wrist=True):
    global wrist_buffer

    if len(landmarks) != 63:
        raise ValueError("Expected 63 landmark values.")
    if len(landmarks_2d) != 21:
        raise ValueError("Expected 21 2D landmarks.")

    normalized = normalize_landmarks(landmarks)
    angle_feats = get_angle_features([tuple(p) for p in landmarks_2d])
    features = normalized + angle_feats
    feature_row = np.array(features).reshape(1, -1)

    probs = model.predict_proba(feature_row)[0]
    confidence = float(max(probs))
    letter = str(model.classes_[np.argmax(probs)]) if confidence > 0.50 else ""

    dtw_letter = None
    if track_wrist:
        # Track wrist for J/Z DTW
        wrist_x = landmarks_2d[0][0]
        wrist_y = landmarks_2d[0][1]
        wrist_buffer.append([wrist_x, wrist_y])

        # Movement guard — suppress I if hand is moving
        hand_is_moving = False
        if len(wrist_buffer) >= 5:
            recent = list(wrist_buffer)[-5:]
            movement = sum(
                abs(recent[i][0] - recent[i-1][0]) + abs(recent[i][1] - recent[i-1][1])
                for i in range(1, len(recent))
            )
            hand_is_moving = movement > 0.02

        if letter == "I" and hand_is_moving:
            letter = ""

        # Check DTW for J/Z
        dtw_letter = check_dtw()

    # Top 5 for confidence bars
    top_indices = np.argsort(probs)[::-1][:5]
    top_predictions = [
        {"letter": str(model.classes_[i]), "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {
        "letter": letter,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "dtw_letter": dtw_letter,
        "normalized": normalized,
        "angle_features": angle_feats,
        "features": features,
    }

 
def check_dtw():
    global dtw_cooldown
    if len(wrist_buffer) < 30:
        return None
    if time.time() < dtw_cooldown:
        return None
 
    wp = np.array(wrist_buffer)

    jdists = []
    for t in jtemplates:
        dx = dtw.distance(wp[:, 0], t[:, 0])
        dy = dtw.distance(wp[:, 1], t[:, 1])
        jdists.append((dx + dy) / 2)
 
    zdists = []
    for t in ztemplates:
        dx = dtw.distance(wp[:, 0], t[:, 0])
        dy = dtw.distance(wp[:, 1], t[:, 1])
        zdists.append((dx + dy) / 2)
 
    best_j = min(jdists) if jdists else float("inf")
    best_z = min(zdists) if zdists else float("inf")
    best   = min(best_j, best_z)
 
    if best < DTW_THRESH:
        dtw_cooldown = time.time() + DTW_COOLDOWN_SECS
        return "J" if best_j < best_z else "Z"
    return None
 
 
@app.post("/predict")
def predict(data: LandmarkData):
    try:
        result = build_prediction(data.landmarks, data.landmarks_2d, track_wrist=True)
 
        return {
            "letter": result["letter"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "dtw_letter": result["dtw_letter"]  # None or "J"/"Z"
        }
 
    except Exception as e:
        return {"letter": "", "confidence": 0.0, "top_predictions": [], "dtw_letter": None, "error": str(e)}


@app.post("/contribute")
async def contribute(
    letter: str = Form(...),
    landmarks: str = Form(...),
    landmarks_2d: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    selected_letter = letter.strip().upper()
    if len(selected_letter) != 1 or selected_letter < "A" or selected_letter > "Z":
        raise HTTPException(status_code=400, detail="Please select a valid A-Z letter.")

    try:
        parsed_landmarks = json.loads(landmarks)
        parsed_landmarks_2d = json.loads(landmarks_2d)
        result = build_prediction(parsed_landmarks, parsed_landmarks_2d, track_wrist=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read contribution landmarks: {exc}") from exc

    predicted_letter = result["letter"]
    confidence = result["confidence"]
    if predicted_letter != selected_letter or confidence < CONTRIBUTE_CONF_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=(
                f'Model predicted "{predicted_letter or "unknown"}" '
                f"({round(confidence * 100)}%) instead of \"{selected_letter}\". "
                "Try better lighting or a clearer angle."
            ),
        )

    contribution_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"
    letter_dir = CONTRIBUTIONS_DIR / selected_letter
    letter_dir.mkdir(parents=True, exist_ok=True)

    image_filename = None
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        image_filename = f"{contribution_id}{suffix}"
        image_path = letter_dir / image_filename
        with image_path.open("wb") as out_file:
            out_file.write(await image.read())

    metadata = {
        "id": contribution_id,
        "letter": selected_letter,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "predicted_letter": predicted_letter,
        "confidence": confidence,
        "top_predictions": result["top_predictions"],
        "landmarks": parsed_landmarks,
        "landmarks_2d": parsed_landmarks_2d,
        "normalized_landmarks": result["normalized"],
        "angle_features": result["angle_features"],
        "features": result["features"],
        "image": image_filename,
    }
    metadata_path = letter_dir / f"{contribution_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "saved": True,
        "id": contribution_id,
        "letter": selected_letter,
        "predicted_letter": predicted_letter,
        "confidence": confidence,
        "path": str(metadata_path),
    }
 
 
@app.post("/reset_wrist")
def reset_wrist():
    """Call this when no hand is detected to clear wrist buffer."""
    wrist_buffer.clear()
    return {"status": "ok"}
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
