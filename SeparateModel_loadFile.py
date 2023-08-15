import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator
from numpy import loadtxt

test_BATCH = 1
VALIDATION_BATCH = 50
SEED = 123
IMSIZE = (100, 100)
data_dir = os.path.join('', 'test')
test_BATCH = 1
#/// Generate the data from data root, structured so that the labeled images are separated in different files ///#
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,)

X_test = test_datagen.flow_from_directory(
    data_dir,
    batch_size=test_BATCH,
    target_size=IMSIZE,
    class_mode='categorical',
    color_mode='grayscale',

)
#/// Save the results in CSV format ///#
predictionsDF = pd.DataFrame(columns=['Predictions'], index=[i for i in range(624)])
model = tf.keras.models.load_model('ready_to_pred.h5')
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
for i, pred in enumerate(model.predict(X_test)):
    predictionsDF['Predictions'].iloc[i] = np.argmax(pred)
predictionsDF.to_csv('predictions.csv')
