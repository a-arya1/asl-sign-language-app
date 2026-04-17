#Process the data using media pipe get the coordinates and store in a CVV file
import os
from wsgiref import headers
import cv2 as cv
import mediapipe as mp
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from normalize_data import normalize_landmarks, get_angle_features
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

mendeley_dataset = "/Users/abhasharyal/Downloads/SignAlphaSet"
mendeley_files = sorted(os.listdir(mendeley_dataset))


with open("handsData.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    dataStorage = []
    for i in range(21):
        dataStorage += [f"x{i}", f"y{i}", f"z{i}"]
    dataStorage.append("thumb_curl")
    dataStorage.append("idx_pip")
    dataStorage.append("idx_dip")
    dataStorage.append("mid_pip")
    dataStorage.append("mid_dip")
    dataStorage.append("ring_pip")
    dataStorage.append("ring_dip")
    dataStorage.append("pinky_pip")
    dataStorage.append("pinky_dip")
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
                xyzValues = normalize_landmarks(xyzValues)

                landmarks = []
                for joint in hands.hand_landmarks[0]:
                    landmarks.append((joint.x, joint.y))
                angleFeatures = get_angle_features(landmarks)

                row = xyzValues + angleFeatures + [files]
                writer.writerow(row)


        print("letter name: " + files)
    for files in mendeley_files:
            if files.isalpha() != True or files.__len__() > 1:
                continue
            folder_path = os.path.join(mendeley_dataset, files)
            if not os.path.isdir(folder_path):
                continue
            for image in os.listdir(folder_path):
                imagesPath = os.path.join(folder_path, image)
                image = cv.imread(imagesPath)
                if image is None:
                    continue
                image_rgb_form = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_form)
                hands = landmarker.detect(mp_image)
                if not hands.hand_landmarks:
                    continue
                else:
                    xyzValues = []
                    for hand in hands.hand_landmarks[0]:
                        xyzValues.append(hand.x)
                        xyzValues.append(hand.y)
                        xyzValues.append(hand.z)
                    xyzValues = normalize_landmarks(xyzValues)

                    landmarks = []
                    for joint in hands.hand_landmarks[0]:
                        landmarks.append((joint.x, joint.y))
                    angleFeatures = get_angle_features(landmarks)

                    row = xyzValues + angleFeatures + [files]
                    writer.writerow(row)
            print("letter name: " + files)



        





