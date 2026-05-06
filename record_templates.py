import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
det = HandLandmarker.create_from_options(options)

os.makedirs('templates/J', exist_ok=True)
os.makedirs('templates/Z', exist_ok=True)

cap = cv2.VideoCapture(0)

# j first then z
for currentLetter in ['J', 'Z']:
    savedSoFar = 0
    print(f"do {currentLetter} now, space to record, q to skip")

    while savedSoFar < 5:
        ret, frm = cap.read()
        if not ret:
            continue
        cv2.putText(frm, f'{currentLetter} ({savedSoFar}/5) space=record', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.imshow('templates', frm)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if k == ord(' '):
            collected = []

            for frameNum in range(30):
                ret, frm = cap.read()
                rgbfrm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbfrm)
                res = det.detect(mpImg)

                if res.hand_landmarks:
                    w = res.hand_landmarks[0][0]
                    collected.append([w.x, w.y])
                else:
                    if len(collected) > 0:
                        collected.append(collected[-1])
                    else:
                        collected.append([0.0, 0.0])

                cv2.putText(frm, f'{frameNum+1}/30', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('templates', frm)
                cv2.waitKey(1)

            np.save(f'templates/{currentLetter}/t{savedSoFar}.npy', np.array(collected))
            savedSoFar += 1
            print(f'got {savedSoFar}')

cap.release()
cv2.destroyAllWindows()