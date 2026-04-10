import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import joblib
from model import predict_sign
model = joblib.load('hand_gesture_model.joblib')

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

current_frame = None
img = None; 

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
        prediction = predict_sign(model, data)


    



def print_current_millisecond_time():
    return round(time.time()*1000)
#Current Time is necessary for tracking


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    result_callback=print_result)

landmarker = HandLandmarker.create_from_options(options)

#Checks if the cameras working
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
    img = np.zeros([height, width, 3], np.uint8)

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

                cv.line(img, (x1,y1), (x2,y2), (255,0,0), 2)

                


    cv.waitKey(2)

    #Get coordinates to display seperate frame of skeleton outline on the first window
    if latest_landmarks and prediction:
                    cv.putText(frame, prediction, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv.LINE_AA)
    cv.imshow('frame', frame)

    
    if img is not None:
        cv.imshow('image', img)
    if cv.waitKey(1) == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()


