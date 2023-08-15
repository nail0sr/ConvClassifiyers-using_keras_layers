import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator
TRAIN_BATCH = 100
VALIDATION_BATCH = 50

VALSPLIT = .20
SEED = 123
IMSIZE = (100, 100)
data_dir = os.path.join('', 'train')
train_normal = os.path.join(data_dir, 'normal')
train_sick = os.path.join(data_dir, 'pneumonia')
TRAIN_BATCH = 100
VALIDATION_BATCH = 50

X_trai_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                             validation_split = 0.2)
X_train =X_trai_gen.flow_from_directory(
    data_dir,
    target_size=(100, 100),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=TRAIN_BATCH,
    subset='training'
)

X_val = X_trai_gen.flow_from_directory(
    data_dir,
    batch_size=20,
    target_size=IMSIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy']) 

hist = model.fit_generator(
    X_train, steps_per_epoch=X_train.samples//TRAIN_BATCH, epochs=10, 
    validation_data=X_val, validation_steps=X_val.samples//20 
)

model.save('ready_to_pred.h5')
