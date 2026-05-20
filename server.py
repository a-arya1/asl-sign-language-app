

"""
server.py - FastAPI backend for ASL prediction with DTW for J/Z
Run with: uvicorn server:app --reload --port 8000
"""
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from normalize_data import normalize_landmarks, get_angle_features
from collections import deque
import time
from dtaidistance import dtw
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
import os
import urllib.request

MODEL_PATH = "hand_gesture_model.joblib"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?export=download&id=1yg_r-LzS1uEODFdtGmgB0snOu5-Wka64"
    urllib.request.urlretrieve(url, MODEL_PATH)

print("Loading model...")
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
    global wrist_buffer
 
    try:
        normalized  = normalize_landmarks(data.landmarks)
        angle_feats = get_angle_features([tuple(p) for p in data.landmarks_2d])
        features    = np.array(normalized + angle_feats).reshape(1, -1)
 
        probs      = model.predict_proba(features)[0]
        confidence = float(max(probs))
        letter     = model.classes_[np.argmax(probs)] if confidence > 0.50 else ""
 
        # Track wrist for J/Z DTW
        wrist_x = data.landmarks_2d[0][0]
        wrist_y = data.landmarks_2d[0][1]
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
            {"letter": model.classes_[i], "confidence": float(probs[i])}
            for i in top_indices
        ]
 
        return {
            "letter": letter,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "dtw_letter": dtw_letter  # None or "J"/"Z"
        }
 
    except Exception as e:
        return {"letter": "", "confidence": 0.0, "top_predictions": [], "dtw_letter": None, "error": str(e)}
 
 
@app.post("/reset_wrist")
def reset_wrist():
    """Call this when no hand is detected to clear wrist buffer."""
    wrist_buffer.clear()
    return {"status": "ok"}
 
 
@app.get("/health")
def health():
    return {"status": "ok"}