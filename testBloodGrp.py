# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Load the trained model
# model = load_model('blood_group_model.h5')

# # Prepare test data from the test image folder
# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     'C:/Users/ASUS/Desktop/BloodGrpDetectionProject/testBLOODGrp',  # Replace with the path to your test image folder
#     target_size=(256, 256),  # Match the input size from training
#     batch_size=32,
#     class_mode='sparse',  # Assuming sparse categorical labels
#     shuffle=False
# )

# # Evaluate the model on the test data
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('blood_group_model.h5')

# Create an instance of ImageDataGenerator for test data with augmentation
test_datagen = ImageDataGenerator(
    rescale=1./255,            
    rotation_range=10,         # Smaller rotation range
    width_shift_range=0.1,     # Smaller width shift range
    height_shift_range=0.1,    # Smaller height shift range
    shear_range=0.1,           # Smaller shear range
    zoom_range=0.1,            # Smaller zoom range
    horizontal_flip=True,      # Flip images horizontally
    fill_mode='nearest'        # Fill missing pixels after transformations
)


# Prepare the test data from the test image folder
test_generator = test_datagen.flow_from_directory(
    'C:/Users/ASUS/Desktop/BloodGrpDetectionProject/testBLOODGrp',  # Replace with the path to your test image folder
    target_size=(256, 256),    # Match the input size from training
    batch_size=32,
    class_mode='sparse',       # Assuming sparse categorical labels
    shuffle=False              # Do not shuffle for proper evaluation
)

# Evaluate the model on the augmented test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
