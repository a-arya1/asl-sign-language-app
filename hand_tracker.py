import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import joblib
from model import predict_sign
from normalize_data import normalize_landmarks, get_angle_features
from collections import deque

model = joblib.load('hand_gesture_model.joblib')
prediction_buffer = deque(maxlen = 7)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
prediction = ""
current_frame = None
current_sentence = []
lastLetter = ""
lastTime = 0
cooldown = 1.5
added_message = ""
added_time = 0
action_message = ""
actionTime = 0
confidence = 0.0
stableLetter = ""
stableLetterCount = 0
stabbleLetterThreshold = 8


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
#Lists of joints which should be connected
HAND_CONNECTIONS_THUMB = [
    (0,1),(1,2),(2,3),(3,4)]
HAND_CONNECTIONS_INDEX = [ 
    (0,5),(5,6),(6,7),(7,8)]
HAND_CONNECTIONS_MIDDLE = [
    (0,9),(9,10),(10,11),(11,12)]
HAND_CONNECTIONS_RING = [
        (0,13),(13,14),(14,15),(15,16)]
HAND_CONNECTIONS_PINKY = [
    (0,17),(17,18),(18,19),(19,20)]
HAND_CONNECTIONS_PALM = [
    (5,9),(9,13),(13,17)]
ALL_CONNECTIONS = [
    HAND_CONNECTIONS_THUMB,
    HAND_CONNECTIONS_INDEX,
    HAND_CONNECTIONS_MIDDLE,
    HAND_CONNECTIONS_RING,
    HAND_CONNECTIONS_PINKY,
    HAND_CONNECTIONS_PALM
]

#Need to track latest parts of hand in order to display the dots
latest_landmarks = []


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_frame
    global prediction 
    global prediction_buffer
    global confidence
    if(current_frame is None):
        return
    height, width, _=  current_frame.shape
    global latest_landmarks
    latest_landmarks=result.hand_landmarks

    #Prints x,y,z coordinates of 21 joints in hand (0 indexed)
    if not(latest_landmarks):
        prediction = ""
        return
    for hand in result.hand_landmarks:
        data = []
        for joint in hand:
            data.append(joint.x)
            data.append(joint.y)
            data.append(joint.z)
        data = normalize_landmarks(data)
        landmarks_2d = []
        for joint in hand:
            landmarks_2d.append((joint.x, joint.y))
        angle_features = get_angle_features(landmarks_2d)
        features = data + angle_features
        
        probs = model.predict_proba(np.array(features).reshape(1, -1))[0]
        prediction_buffer.append(probs)

        avg = np.mean(prediction_buffer, axis=0)
        confidence = float(max(avg))
        if confidence > 0.50:
            prediction = model.classes_[np.argmax(avg)]
        else:
            prediction = ""


        global current_sentence, lastLetter, lastTime, added_message, added_time
        global stableLetter, stableLetterCount

        if prediction:
            currTime = time.time()
            if prediction == stableLetter:
                stableLetterCount += 1
            else:
                stableLetter = prediction
                stableLetterCount = 1  # reset count if letter changed

            if stableLetterCount == stabbleLetterThreshold:
                if prediction != lastLetter or (currTime - lastTime) > cooldown:
                    current_sentence.append(prediction)
                    lastLetter = prediction
                    lastTime = currTime
                    added_message = f"Added: {prediction}"
                    added_time = currTime
        else:
            stableLetter = ""
            stableLetterCount = 0

        break


    



def print_current_millisecond_time():
    return round(time.time()*1000)
#Current Time is necessary for tracking


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    result_callback=print_result)

landmarker = HandLandmarker.create_from_options(options)

#Chec ks if the cameras working
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    #Capture all frames of the video 
    ret, frame = cap.read()
    #Is the frame is read correctly ret is true
    if not ret:
        print("Unable to recieve frame")
        break
    #Convert frame from BGR to RGB
    RGBFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGBFrame)
    landmarker.detect_async(mp_image, print_current_millisecond_time())

    #Shows Frame
    current_frame = frame
    height, width, _=  current_frame.shape

    for hand in latest_landmarks:
        for index, joint in enumerate(hand):
            x= int(joint.x * width)
            y = int(joint.y * height)
            cv.circle(frame, (x,y), 5, (0,255,0), -1)
        for finger in ALL_CONNECTIONS:
            for coordinatePair in finger:
                #Skeleton Outline of Hand
                startingCoord = coordinatePair[0]
                endingCoord = coordinatePair[1]
                handPositionStart = hand[startingCoord]
                handPositionEnd = hand[endingCoord]
                x1 = int(handPositionStart.x * width)
                y1 = int(handPositionStart.y * height)
                x2 = int(handPositionEnd.x * width)
                y2 = int(handPositionEnd.y * height)
                cv.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)


                


    
    #Some of this UI stuff below was made by claude
    currentTime = time.time()
    
    box_height = 120
    box_start_y = height - box_height
    
    overlay = frame.copy()
    cv.rectangle(overlay, (0, box_start_y), (width, height), (40, 40, 40), -1)
    cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv.rectangle(frame, (0, box_start_y), (width, height), (100, 100, 100), 2)
    current_sentence_text = "".join(current_sentence)
    if int(currentTime * 2) % 2 == 0: 
        current_sentence_text += "|"
    
    cv.putText(frame, current_sentence_text, (20, box_start_y + 65), 
               cv.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    
    # Label
    cv.putText(frame, "Your Message:", (20, box_start_y + 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    
    if prediction:
        letter_text = f"{prediction}"
        text_size = cv.getTextSize(letter_text, cv.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        text_x = (width - text_size[0]) // 2
        
        cv.circle(frame, (text_x + text_size[0]//2, 120), 60, (0, 180, 0), -1)
        cv.circle(frame, (text_x + text_size[0]//2, 120), 60, (0, 255, 0), 3)
        
        cv.putText(frame, letter_text, (text_x, 140), 
                   cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
        cv.putText(frame, f"{int(confidence * 100)}%", (text_x, 175),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    else:
        cv.putText(frame, "No letter detected", (width//2 - 150, 130), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    
    if currentTime - added_time < 0.5:
        text_size = cv.getTextSize(added_message, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width - text_size[0] - 20
        cv.putText(frame, added_message, (text_x, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    
    if currentTime - actionTime < 1.0:
        cv.putText(frame, action_message, (20, box_start_y - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    
    
    instructions = [
        "SPACE = space",
        "BACKSPACE = Delete",
        "C = Clear",
        "Q = Quit"
    ]
    
    inst_box_width = 280
    inst_box_height = len(instructions) * 30 + 20
    overlay_inst = frame.copy()
    cv.rectangle(overlay_inst, (10, 10), (inst_box_width, inst_box_height), (30, 30, 30), -1)
    cv.addWeighted(overlay_inst, 0.6, frame, 0.4, 0, frame)
    cv.rectangle(frame, (10, 10), (inst_box_width, inst_box_height), (80, 80, 80), 1)
    
    y_position = 35
    for instruction in instructions:
        cv.putText(frame, instruction, (20, y_position), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_position += 30
    cv.imshow('frame', frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        current_sentence = []
        lastLetter = ""
        action_message = "cleared!"    
        actionTime = time.time()
    elif key == 32:
        current_sentence.append(" ")
        lastLetter = " "
        lastTime = time.time()
        added_message = "Added: Space"
        added_time = time.time()
    elif key == 0 or key == 127:
        removed = current_sentence.pop()
        action_message = f"Removed: {removed}"
        actionTime = time.time()

cap.release()
cv.destroyAllWindows()


