import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
#Talk about why using random forest instead of tensor flow 
#better for tabular data, tensor flow would be overkill. 


def predict_sign(model, landmark_data):
    landmark_data = np.array(landmark_data).reshape(1, -1)
    proba = model.predict_proba(landmark_data)[0]
    mostConfident = max(proba)
    if mostConfident > 0.2:
        predictedletter = model.classes_[np.argmax(proba)]
        return predictedletter
    else:
        return ""

    return model.predict(landmark_data)[0]

#use this function later in hand_tracker.py with the landmark data

dataFrame = pd.read_csv('handsData.csv')
#loads csv as a table
#Talk about why a dataframe is used here for presentation
#its faster than other data structures for tabular data, almost all libraries are used with them
label = dataFrame.columns[-1] #Gets label which is last column in the csv (the letter)
num_classes = len(dataFrame[label].unique())

x = dataFrame.drop(columns=[label])
y = dataFrame[label]
#use 20% of the data for testing the model and 80% for training it.
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2, random_state=99) #random seed number easier to debug
model = RandomForestClassifier(n_estimators=200, random_state=99, n_jobs=-1)
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy}")
print(classification_report(yTest, yPred))

joblib.dump(model, 'hand_gesture_model.joblib')

