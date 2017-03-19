import csv
import cv2
import numpy as np

# load data
lines = []
#for directory  in ('data', 'data1'):
for directory  in ('data1',):
    with open('data1/driving_log.csv') as f:
        lines += [line for line in csv.reader(f)]

correction = 0.2 # this is a parameter to tune

# original data
images = [cv2.imread(line[0]) for line in lines] + [cv2.imread(line[1]) for line in lines] + [cv2.imread(line[2]) for line in lines]
angle = [float(line[3]) for line in lines] + [float(line[3])+correction for line in lines] + [float(line[3])-correction for line in lines]
throttle = [float(line[4]) for line in lines] + [float(line[4]) for line in lines] + [float(line[4]) for line in lines]

# column flipped data
images = images + [cv2.flip(image, 1) for image in images]
angle = angle + [-x for x in angle]
throttle = throttle + throttle

X_train = np.array(images)
y_train = np.array(list(zip(angle, throttle)))
print (X_train.shape, y_train.shape)

# define the model
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24, kernel_size=(5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(36, kernel_size=(5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(48, kernel_size=(5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
#model.add(Dense(10))
model.add(Dense(y_train.shape[1]))
model.compile(optimizer='adam', loss='mse')
plot_model(model, to_file='model.png', show_shapes=True)

# fit the model
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')

# plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('error.png')
