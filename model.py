import pandas as pd
import numpy as np


dataFrame = pd.read_csv('handsData.csv')
#loads csv as a table
#Talk about why a dataframe is used here for presentation
#its faster than other data structures for tabular data, almost all libraries are used with them
label = dataFrame.columns[-1] #Gets label which is last column in the csv (the letter)
num_classes = len(dataFrame[label].unique())

x = dataFrame.drop(columns=[label])
y = dataFrame[label]

print(f"features {x.shape}")
print(f"target shape {y.shape}")