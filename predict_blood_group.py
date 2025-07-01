from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('blood_group_model.h5')

# Load the external BMP image and preprocess it
img = image.load_img(r'C:\Users\ASUS\Desktop\BloodGrpDetectionProject\testBLOODGrp\A+\cluster_0_5993.BMP', target_size=(256,256))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Scale the image

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Map the prediction to the blood group label
class_labels = {0: 'A-', 1: 'A+', 2: 'AB-', 3: 'AB+', 4: 'B-', 5: 'B+', 6: 'O-', 7: 'O+'}

# Output the predicted blood group
predicted_blood_group = class_labels[predicted_class]
print(f'Predicted Blood Group: {predicted_blood_group}')


