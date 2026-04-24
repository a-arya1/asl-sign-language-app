import cv2 as cv
import mediapipe as mp
import csv
import os
from normalize_data import normalize_landmarks, get_angle_features

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

OUTPUT_CSV = "handsData.csv"
counts = {chr(i): 0 for i in range(ord('A'), ord('Z')+1)}
currentLetter = None
collecting = False

cap = cv.VideoCapture(0)

# check if file exists already so we dont overwrite headers
fileExists = os.path.exists(OUTPUT_CSV)

with open(OUTPUT_CSV, mode="a", newline="") as file:
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        handDetected = len(result.hand_landmarks) > 0

        # draw landmarks
        if handDetected:
            h, w, _ = frame.shape
            for joint in result.hand_landmarks[0]:
                cx, cy = int(joint.x * w), int(joint.y * h)
                cv.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # UI
        cv.putText(frame, "Press a letter key to start collecting that letter", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv.putText(frame, "Hold sign steady - auto collects while hand detected",
                   (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv.putText(frame, "Q = quit",
                   (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if currentLetter:
            color = (0,255,0) if handDetected else (0,0,255)
            # big text in center of screen
            text = f"{currentLetter}: {counts[currentLetter]} samples"
            textSize = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
            textX = (frame.shape[1] - textSize[0]) // 2
            cv.putText(frame, text, (textX, frame.shape[0] // 2),
                    cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            if not handDetected:
                cv.putText(frame, "No hand detected - reposition hand",
                        (10, frame.shape[0] // 2 + 50),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            # prompt before any letter is selected
            textSize = cv.getTextSize("Press a letter key to begin", cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = (frame.shape[1] - textSize[0]) // 2
            cv.putText(frame, "Press a letter key to begin", (textX, frame.shape[0] // 2),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            if not handDetected:
                cv.putText(frame, "No hand detected", (10, 180),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # show per-letter counts in corner
        y_pos = 160
        cv.putText(frame, "Collected:", (10, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        y_pos += 20
        for letter, count in counts.items():
            if count > 0:
                cv.putText(frame, f"{letter}: {count}", (10, y_pos),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (100,255,100), 1)
                y_pos += 18

        cv.imshow("Collect Data", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('x'):
            break

        # letter key pressed - switch current letter
        if 97 <= key <= 122:  # a-z
            currentLetter = chr(key - 32)  # convert to uppercase
            print(f"Switched to letter: {currentLetter}")

        # auto collect whenever hand is detected and a letter is selected
        if currentLetter and handDetected:
            vals = []
            for joint in result.hand_landmarks[0]:
                vals.append(joint.x)
                vals.append(joint.y)
                vals.append(joint.z)
            normalized = normalize_landmarks(vals)
            lm_2d = [(joint.x, joint.y) for joint in result.hand_landmarks[0]]
            angles = get_angle_features(lm_2d)
            row = normalized + angles + [currentLetter]
            writer.writerow(row)
            counts[currentLetter] += 1

cap.release()
cv.destroyAllWindows()
print("\nFinal counts:")
for letter, count in counts.items():
    print(f"  {letter}: {count}")