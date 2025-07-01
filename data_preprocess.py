import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle

############################################################
# Folder location and categories
dir = r'C:\Users\ASUS\Desktop\BloodGrpDetectionProject\dataset_blood_group'
category = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
IMG_SIZE = 256
data = []

# Iterate over each category
for cat in category:
    folder = os.path.join(dir, cat)
    print(f"Looking in folder: {folder}")  # Debug print statement
    label = category.index(cat)
    
    # Check if the folder exists
    if os.path.exists(folder):
        # Iterate over each image in the folder
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            
            # Read and process the image
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
            data.append([img_arr, label])
    else:
        print(f"Folder not found: {folder}")  # Error message if the folder doesn't exist

# Shuffle the data to ensure it's not ordered
random.shuffle(data)

# Splitting the data into features and labels
x = []
y = []
for features, label in data:
    x.append(features)
    y.append(label)

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)

# Save the data using pickle
pickle.dump(x, open('x.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
