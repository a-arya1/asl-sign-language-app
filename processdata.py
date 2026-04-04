#Process the data using media pipe get the coordinates and store in a CVV file
import os
import cv2 as cv
import mediapipe as mp
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands = 1,
)

landmarker = HandLandmarker.create_from_options(options)
#Create CSV File with landmark data for each letter
dataset = "archive/asl_alphabet_train/asl_alphabet_train"
dataset_files = sorted(os.listdir(dataset))

    

with open("handsData.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    dataStorage = []
    for i in range(21):
        dataStorage += [f"x{i}", f"y{i}", f"z{i}"]
    
    dataStorage.append("Letter Label")
    writer.writerow(dataStorage)
    #add labels
    for files in dataset_files:
        if files.isalpha() != True or files.__len__() > 1:
            continue
        folder_path = os.path.join(dataset, files)
#loop through dataset and add coordinates for each letter into the csv
        for image in os.listdir(folder_path):
            imagesPath = os.path.join(folder_path, image)
            image = cv.imread(imagesPath)
            if(image is None):
                continue
            image_rgb_form = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_form)
            hands = landmarker.detect(mp_image)
            if not hands.hand_landmarks:
                continue
            else:
                #The dataset only has one hand so you can just loop through the first one
                xyzValues = []
                for hand in hands.hand_landmarks[0]:
                    xyzValues.append(hand.x)
                    xyzValues.append(hand.y)
                    xyzValues.append(hand.z)
                xyzValues.append(files)
                writer.writerow(xyzValues)
        print("letter name: " + files)




        





