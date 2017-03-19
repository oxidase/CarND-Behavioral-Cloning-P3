import csv
import cv2
import numpy as np

with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    lines = [line for line in reader]

images = [cv2.imread(line[0]) for line in lines]
measurements = [line[3] for line in lines]

X_train = np.array(images)
y_train = np.array(measurements)
print(measurements)

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
