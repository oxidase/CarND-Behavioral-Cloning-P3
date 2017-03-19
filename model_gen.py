import csv
import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# load data
def load_csv(directories, correction = 0.2):
    lines = []
    for directory  in directories:
        with open(directory + '/driving_log.csv') as f:
            lines += [line for line in csv.reader(f)]

    # original data
    images = [line[0] for line in lines] + [line[1] for line in lines] + [line[2] for line in lines]
    angle = [float(line[3]) for line in lines] + [float(line[3])+correction for line in lines] + [float(line[3])-correction for line in lines]
    throttle = [float(line[4]) for line in lines] + [float(line[4]) for line in lines] + [float(line[4]) for line in lines]

    # add columns-flipped data
    images = [(image, False) for image in images] + [(image, True) for image in images]
    angle = angle + [-x for x in angle]
    throttle = throttle + throttle

    return list(zip(images, angle, throttle))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images  = [cv2.flip(cv2.imread(x[0][0]), 1) if x[0][1] else cv2.imread(x[0][0]) for x in batch_samples]
            angles = [x[1] for x in batch_samples]
            throttle = [x[2] for x in batch_samples]

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


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
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
plot_model(model, to_file='model.png', show_shapes=True)


# train the model using the generator function
samples = load_csv(('data', 'data1', 'data_out', 'data_rev'))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / batch_size, \
                    validation_data=validation_generator, validation_steps=len(validation_samples) / batch_size, \
                    epochs=7)

model.save('model.h5')

import matplotlib.pyplot as plt
print (history_object)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('error.png')
#plt.show()
