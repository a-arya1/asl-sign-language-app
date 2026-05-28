"""
server.py - FastAPI backend for ASL prediction with DTW for J/Z
Run with: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
import csv
import uuid
import glob

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
CONTRIBUTION_ROWS_PATH = Path("contributions/training_rows.csv")
PREDICT_CONF_THRESHOLD = 0.45
CONTRIB_SECRET = "asl-token-8472xq"
CONTRIBUTE_CONF_THRESHOLD = 0.50
PREDICTION_SMOOTHING_FRAMES = 5
ADAPTIVE_MAX_DISTANCE = 0.42
ADAPTIVE_BLEND_CONFIDENCE = 0.80

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    import gdown
    gdown.download(id="1yg_r-LzS1uEODFdtGmgB0snOu5-Wka64", output=MODEL_PATH, quiet=False)
    print("Download complete.")
model = joblib.load(MODEL_PATH)
print(f"Model loaded. Classes: {list(model.classes_)}")

# Load J/Z templates
def normalize_motion_path(path):
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path
    path = path - path[0]
    scale = np.max(np.ptp(path, axis=0))
    if scale > 1e-6:
        path = path / scale
    return path


def load_templates(letter):
    templates = []
    for path in sorted(glob.glob(f"templates/{letter}/*.npy")):
        try:
            t = np.load(path)
            templates.append(normalize_motion_path(t))
        except (OSError, ValueError):
            pass
    print(f"Loaded {len(templates)} templates for {letter}")
    return templates

jtemplates = load_templates("J")
ztemplates = load_templates("Z")

# Per-client state (single user assumed for local use)
wrist_buffer = deque(maxlen=30)
prob_buffer = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
contribution_examples = []
dtw_cooldown = 0
DTW_COOLDOWN_SECS = 2.0
DTW_READY_FRAMES = 24
DTW_THRESH = 0.90
DTW_MARGIN = 0.18
DTW_MIN_MOVEMENT = 0.08


class LandmarkData(BaseModel):
    landmarks: list       # flat [x,y,z ...] 63 values
    landmarks_2d: list    # [[x,y] ...] 21 pairs
    wrist_path: Optional[list] = None


def scaled_feature_vector(features):
    values = np.asarray(features, dtype=float)
    if values.shape[0] >= 72:
        values = values.copy()
        values[63:72] = values[63:72] / 180.0
    return values


def load_contribution_examples():
    examples = []
    for metadata_path in CONTRIBUTIONS_DIR.glob("*/*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            features = metadata.get("features")
            letter = metadata.get("letter")
            if letter and features and len(features) == 72:
                examples.append({
                    "letter": letter,
                    "features": scaled_feature_vector(features),
                    "path": str(metadata_path),
                })
        except (OSError, json.JSONDecodeError):
            continue
    return examples


def nearest_contribution(features):
    if not contribution_examples:
        return None
    current = scaled_feature_vector(features)
    best = None
    for example in contribution_examples:
        distance = float(np.linalg.norm(current - example["features"]) / np.sqrt(len(current)))
        if best is None or distance < best["distance"]:
            best = {**example, "distance": distance}
    return best


def apply_adaptive_prediction(letter, confidence, top_predictions, features):
    nearest = nearest_contribution(features)
    if not nearest or nearest["distance"] > ADAPTIVE_MAX_DISTANCE:
        return letter, confidence, top_predictions, None
    adaptive_letter = nearest["letter"]
    top_letters = [p["letter"] for p in top_predictions[:3]]
    if confidence >= ADAPTIVE_BLEND_CONFIDENCE and letter and letter != adaptive_letter:
        return letter, confidence, top_predictions, nearest
    if letter and adaptive_letter not in top_letters and letter != adaptive_letter:
        return letter, confidence, top_predictions, nearest

    boosted_confidence = max(confidence, ADAPTIVE_BLEND_CONFIDENCE - nearest["distance"] * 0.25)
    updated_top = [{"letter": adaptive_letter, "confidence": boosted_confidence}]
    for item in top_predictions:
        if item["letter"] != adaptive_letter:
            updated_top.append(item)
        if len(updated_top) == 5:
            break
    return adaptive_letter, boosted_confidence, updated_top, nearest


def append_training_row(features, letter):
    CONTRIBUTION_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CONTRIBUTION_ROWS_PATH.exists()
    with CONTRIBUTION_ROWS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            header = []
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            header += [
                "thumb_curl",
                "idx_pip",
                "idx_dip",
                "mid_pip",
                "mid_dip",
                "ring_pip",
                "ring_dip",
                "pinky_pip",
                "pinky_dip",
                "Letter Label",
            ]
            writer.writerow(header)
        writer.writerow(list(features) + [letter])


contribution_examples = load_contribution_examples()
print(f"Loaded {len(contribution_examples)} contribution examples")


def build_prediction(landmarks, landmarks_2d, wrist_path=None, track_wrist=True, smooth=True, adaptive=True):
    global wrist_buffer, prob_buffer

    if len(landmarks) != 63:
        raise ValueError("Expected 63 landmark values.")
    if len(landmarks_2d) != 21:
        raise ValueError("Expected 21 2D landmarks.")

    normalized = normalize_landmarks(landmarks)
    angle_feats = get_angle_features([tuple(p) for p in landmarks_2d])
    features = normalized + angle_feats
    feature_row = np.array(features).reshape(1, -1)

    probs = model.predict_proba(feature_row)[0]
    if smooth:
        prob_buffer.append(probs)
        probs = np.mean(prob_buffer, axis=0)

    confidence = float(max(probs))
    letter = str(model.classes_[np.argmax(probs)]) if confidence > PREDICT_CONF_THRESHOLD else ""

    dtw_letter = None
    if track_wrist:
        if wrist_path and len(wrist_path) >= DTW_READY_FRAMES:
            wrist_buffer.clear()
            wrist_buffer.extend(wrist_path[-wrist_buffer.maxlen:])
        else:
            wrist_x = landmarks_2d[0][0]
            wrist_y = landmarks_2d[0][1]
            wrist_buffer.append([wrist_x, wrist_y])

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

        dtw_letter = check_dtw()

    top_indices = np.argsort(probs)[::-1][:5]
    top_predictions = [
        {"letter": str(model.classes_[i]), "confidence": float(probs[i])}
        for i in top_indices
    ]

    adaptive_match = None
    if adaptive:
        letter, confidence, top_predictions, adaptive_match = apply_adaptive_prediction(
            letter,
            confidence,
            top_predictions,
            features,
        )

    return {
        "letter": letter,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "dtw_letter": dtw_letter,
        "normalized": normalized,
        "angle_features": angle_feats,
        "features": features,
        "adaptive_match": adaptive_match,
    }


def check_dtw():
    global dtw_cooldown
    if len(wrist_buffer) < DTW_READY_FRAMES:
        return None
    if time.time() < dtw_cooldown:
        return None

    raw_wp = np.array(wrist_buffer)
    movement = np.sum(np.linalg.norm(np.diff(raw_wp, axis=0), axis=1))
    if movement < DTW_MIN_MOVEMENT:
        return None
    wp = normalize_motion_path(raw_wp)

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
    best = min(best_j, best_z)
    second = max(best_j, best_z)

    if best < DTW_THRESH and (second - best) >= DTW_MARGIN:
        dtw_cooldown = time.time() + DTW_COOLDOWN_SECS
        return "J" if best_j < best_z else "Z"
    return None


@app.post("/predict")
def predict(data: LandmarkData):
    try:
        result = build_prediction(data.landmarks, data.landmarks_2d, data.wrist_path, track_wrist=True)
        return {
            "letter": result["letter"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "dtw_letter": result["dtw_letter"]
        }
    except Exception as e:
        return {"letter": "", "confidence": 0.0, "top_predictions": [], "dtw_letter": None, "error": str(e)}


@app.post("/contribute")
async def contribute(
    request: Request,
    letter: str = Form(...),
    landmarks: str = Form(...),
    landmarks_2d: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    token = request.headers.get("X-Contrib-Token", "")
    if token != CONTRIB_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized.")

    selected_letter = letter.strip().upper()
    if len(selected_letter) != 1 or selected_letter < "A" or selected_letter > "Z":
        raise HTTPException(status_code=400, detail="Please select a valid A-Z letter.")

    try:
        parsed_landmarks = json.loads(landmarks)
        parsed_landmarks_2d = json.loads(landmarks_2d)
        result = build_prediction(
            parsed_landmarks,
            parsed_landmarks_2d,
            track_wrist=False,
            smooth=False,
            adaptive=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read contribution landmarks: {exc}") from exc

    contribution_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"
    letter_dir = CONTRIBUTIONS_DIR / selected_letter
    letter_dir.mkdir(parents=True, exist_ok=True)

    image_filename = None
    if image and image.filename:
        contents = await image.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Please use an image under 5MB.")
        await image.seek(0)
        suffix = Path(image.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        image_filename = f"{contribution_id}{suffix}"
        image_path = letter_dir / image_filename
        with image_path.open("wb") as out_file:
            out_file.write(contents)

    metadata = {
        "id": contribution_id,
        "letter": selected_letter,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "predicted_letter": selected_letter,
        "confidence": 1.0,
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
    append_training_row(result["features"], selected_letter)
    contribution_examples.append({
        "letter": selected_letter,
        "features": scaled_feature_vector(result["features"]),
        "path": str(metadata_path),
    })

    return {
        "saved": True,
        "trained": True,
        "training_mode": "adaptive_contribution_memory",
        "id": contribution_id,
        "letter": selected_letter,
        "predicted_letter": selected_letter,
        "confidence": 1.0,
        "angle_features": result["angle_features"],
        "path": str(metadata_path),
    }


@app.post("/reset_wrist")
def reset_wrist():
    wrist_buffer.clear()
    prob_buffer.clear()
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "contribution_examples": len(contribution_examples)}


@app.get("/training_status")
def training_status():
    return {
        "adaptive_examples": len(contribution_examples),
        "training_rows_file": str(CONTRIBUTION_ROWS_PATH),
        "training_rows_file_exists": CONTRIBUTION_ROWS_PATH.exists(),
    }