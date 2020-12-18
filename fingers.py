import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from skimage import io, transform

import os, glob

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


train_images = glob.glob("train/*.png")
test_images = glob.glob("test/*.png")

X_train = []
X_test = []
y_train = []
y_test = []
    
for img in train_images:
    img_read = cv2.imread(img)
    # Most images are already of size (128,128) but it is always better to ensure they all are
    img_read = cv2.resize(img_read, (128,128))
    X_train.append(img_read)
    # The info about the number of fingers and the fact that this is a right or left hand is in two characters of the path
    y_train.append(img[-6:-4])
    
for img in test_images:
    img_read = cv2.imread(img)
    img_read = cv2.resize(img_read, (128,128))
    X_test.append(img_read)
    y_test.append(img[-6:-4])

print ("Shape of an image in X_train: ", X_train[0].shape)
print ("Shape of an image in X_test: ", X_test[0].shape)


from sklearn import preprocessing
import tensorflow as tf
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=12)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=12)

y_train = np.array(y_train)
X_train = np.array(X_train)

y_test = np.array(y_test)
X_tese = np.array(X_test)



print ("Shape of an image in X_train: ", X_train[0].shape)
print ("Shape of an image in X_test: ", X_test[0].shape)
print("Total categories: ", len(np.unique(y_train)))
print("Total categories: ", len(np.unique(y_test)))


print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_tese.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128,128, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=12, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


cnn.fit(x = X_train,y = y_train, batch_size=64, validation_data = (X_tese,y_test), epochs = 10)







