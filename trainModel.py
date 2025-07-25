# # # import pickle
# # # import time
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# # # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # from tensorflow.keras.callbacks import TensorBoard

# # # NAME=f'detect-blood-group-{int(time.time())}'

# # # tensorboard=TensorBoard(log_dir=f'logs\\{NAME}\\')

# # # # Load the data
# # # X = pickle.load(open('x.pkl', 'rb'))
# # # y = pickle.load(open('y.pkl', 'rb'))

# # # # Feature scaling for faster calculation
# # # X = X / 255.0

# # # # Define the model
# # # model = Sequential()
# # # model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
# # # model.add(MaxPooling2D((2, 2)))
# # # model.add(Flatten())
# # # model.add(Dense(128, activation='relu'))
# # # model.add(Dense(8, activation='softmax'))  # 8 classes for blood groups

# # # # Compile the model
# # # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # # # Use ImageDataGenerator to avoid memory overload
# # # datagen = ImageDataGenerator(validation_split=0.1)

# # # train_generator = datagen.flow(X, y, batch_size=32, subset='training')
# # # validation_generator = datagen.flow(X, y, batch_size=32, subset='validation')

# # # # Train the model
# # # model.fit(train_generator, epochs=5, validation_data=validation_generator,callbacks=[tensorboard])

# # # # Save the trained model
# # # model.save('blood_group_model.h5')
# # # print("Model saved as 'blood_group_model.h5'")
# # import pickle
# # import time
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, classification_report
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.callbacks import TensorBoard

# # NAME = f'detect-blood-group-{int(time.time())}'

# # tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

# # # Load the data
# # X = pickle.load(open('x.pkl', 'rb'))
# # y = pickle.load(open('y.pkl', 'rb'))

# # # Feature scaling for faster calculation
# # X = X / 255.0

# # # Split the data: 80% training, 20% testing
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Define the model
# # model = Sequential()
# # model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
# # model.add(MaxPooling2D((2, 2)))
# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # model.add(Dense(8, activation='softmax'))  # 8 classes for blood groups

# # # Compile the model
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # # Use ImageDataGenerator to avoid memory overload
# # datagen = ImageDataGenerator()

# # train_generator = datagen.flow(X_train, y_train, batch_size=16)
# # test_generator = datagen.flow(X_test, y_test, batch_size=16)

# # # Train the model
# # model.fit(train_generator, epochs=5, validation_data=test_generator, callbacks=[tensorboard])

# # # Save the trained model
# # model.save('blood_group_model.h5')
# # print("Model saved as 'blood_group_model.h5'")

# # # Predict on the test set
# # y_pred = model.predict(X_test)
# # y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# # # Confusion matrix
# # conf_matrix = confusion_matrix(y_test, y_pred_classes)
# # print("Confusion Matrix:")
# # print(conf_matrix)

# # # Classification report for more metrics (precision, recall, F1-score)
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred_classes))  # Move this up before plotting the confusion matrix

# # # Plot confusion matrix
# # plt.figure(figsize=(10, 7))
# # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# # plt.title("Confusion Matrix")
# # plt.ylabel("Actual Class")
# # plt.xlabel("Predicted Class")
# # plt.show()


# import pickle
# import time
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import TensorBoard

# # TensorBoard setup
# NAME = f'detect-blood-group-{int(time.time())}'
# tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

# # Load the data
# X = pickle.load(open('x.pkl', 'rb'))
# y = pickle.load(open('y.pkl', 'rb'))

# # Feature scaling for faster calculation
# X = X / 255.0

# # Split the data: 80% training, 20% testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the model
# model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(8, activation='softmax'))  # 8 classes for blood groups

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Use ImageDataGenerator to avoid memory overload
# datagen = ImageDataGenerator()

# train_generator = datagen.flow(X_train, y_train, batch_size=16)
# test_generator = datagen.flow(X_test, y_test, batch_size=16)

# # Train the model
# model.fit(train_generator, epochs=5, validation_data=test_generator, callbacks=[tensorboard])

# # Save the trained model
# model.save('blood_group_model.h5')
# print("Model saved as 'blood_group_model.h5'")

# # Predict on the test set
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# # Confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred_classes)

# # Classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred_classes))

# # Plot confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.ylabel("Actual Class")
# plt.xlabel("Predicted Class")
# plt.show()


import pickle
import time
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# TensorBoard setup
NAME = f'detect-blood-group-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

# Load the data
X = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

# Feature scaling for faster calculation
X = X / 255.0

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 8 classes for blood groups

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for loading data without augmentation to avoid random transformations
datagen = ImageDataGenerator()

# No shuffling to keep data order consistent
train_generator = datagen.flow(X_train, y_train, batch_size=16, shuffle=False)
test_generator = datagen.flow(X_test, y_test, batch_size=16, shuffle=False)

# Train the model
model.fit(train_generator, epochs=10,
          validation_data=test_generator, callbacks=[tensorboard])

# Save the trained model
model.save('blood_group_model.h5')
print("Model saved as 'blood_group_model.h5'")

# Predict on the test set
y_pred = model.predict(X_test)
# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.show()

# test comment
