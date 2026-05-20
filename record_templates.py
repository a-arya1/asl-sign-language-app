"""
record_templates.py
-------------------
Run this ONCE locally to record your personal J and Z signing templates.
These are saved as .npy files and uploaded alongside index.html.

Usage:
    python record_templates.py

Controls:
    SPACE  = start recording a clip (records next 30 frames automatically)
    S      = save the completed clip as a template
    R      = discard and re-record
    N      = skip to next letter
    Q      = quit early

Output:
    templates/J/template_0.npy ... template_9.npy
    templates/Z/template_0.npy ... template_9.npy

Each .npy file is a (30, 2) array of normalized wrist (x, y) positions.
Upload the entire templates/ folder to your GitHub repo alongside index.html.
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import platform

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

TEMPLATE_FRAMES = 30
TEMPLATES_PER_LETTER = 10
LETTERS = ["J", "Z"]


def open_camera():
    system = platform.system()
    if system == "Darwin":
        backends = [cv.CAP_AVFOUNDATION]
    elif system == "Windows":
        backends = [cv.CAP_DSHOW, cv.CAP_MSMF]
    elif system == "Linux":
        backends = [cv.CAP_V4L2]
    else:
        backends = [cv.CAP_ANY]
    for backend in backends:
        for index in range(5):
            cap = cv.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Using camera {index}")
                return cap
            cap.release()
    raise RuntimeError("No working camera found")


def get_wrist_pos(frame):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.hand_landmarks:
        return None
    wrist = result.hand_landmarks[0][0]
    return (wrist.x, wrist.y)


def record_letter(cap, letter, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    count = len(existing)
    recording = False
    clip = []
    status = ""

    print(f"\n── Recording templates for {letter} ──")
    print(f"Already have {count}/{TEMPLATES_PER_LETTER} templates.")

    while count < TEMPLATES_PER_LETTER:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape

        wrist = get_wrist_pos(frame)

        if wrist:
            cx, cy = int(wrist[0] * w), int(wrist[1] * h)
            cv.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

        if recording:
            cv.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
            cv.putText(frame, f"Recording... {len(clip)}/{TEMPLATE_FRAMES}",
                       (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if wrist:
                clip.append(wrist)
            if len(clip) >= TEMPLATE_FRAMES:
                recording = False
                status = f"Clip complete! Press S to save or R to redo."

        cv.putText(frame, f"Letter: {letter}  Saved: {count}/{TEMPLATES_PER_LETTER}",
                   (10, h - 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(frame, "SPACE=record  S=save  R=redo  N=next letter  Q=quit",
                   (10, h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if status:
            cv.putText(frame, status, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        if not wrist:
            cv.putText(frame, "No hand detected", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if clip:
            for i in range(1, len(clip)):
                p1 = (int(clip[i-1][0] * w), int(clip[i-1][1] * h))
                p2 = (int(clip[i][0] * w), int(clip[i][1] * h))
                cv.line(frame, p1, p2, (0, 200, 255), 2)

        cv.imshow("Record Templates", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            return False
        elif key == 32:
            clip = []
            recording = True
            status = ""
        elif key == ord('r'):
            clip = []
            recording = False
            status = "Cleared. Press SPACE to try again."
        elif key == ord('s'):
            if len(clip) == TEMPLATE_FRAMES:
                path = os.path.join(save_dir, f"template_{count}.npy")
                np.save(path, np.array(clip))
                count += 1
                print(f"  Saved {path}  ({count}/{TEMPLATES_PER_LETTER})")
                clip = []
                status = f"Saved! {count}/{TEMPLATES_PER_LETTER} templates done."
            else:
                status = "No complete clip yet. Press SPACE to record."
        elif key == ord('n'):
            print(f"  Skipping {letter} with {count} templates.")
            return True

    print(f"  Done! {count} templates saved for {letter}.")
    return True


cap = open_camera()

for letter in LETTERS:
    save_dir = os.path.join("templates", letter)
    if not record_letter(cap, letter, save_dir):
        print("Quitting early.")
        break

cap.release()
cv.destroyAllWindows()

print("\nDone! Upload your templates/ folder to GitHub alongside index.html.")
print("  templates/J/template_0.npy  ...  template_9.npy")
print("  templates/Z/template_0.npy  ...  template_9.npy")
