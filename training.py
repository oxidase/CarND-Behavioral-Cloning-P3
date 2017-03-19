import csv
import cv2
import numpy as np

with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    lines = [line for line in reader]

correction = 0.2 # this is a parameter to tune

# original data
images = [cv2.imread(line[0]) for line in lines] + [cv2.imread(line[1]) for line in lines] + [cv2.imread(line[2]) for line in lines]
measurements = [float(line[3]) for line in lines] + [float(line[3])+correction for line in lines] + [float(line[3])-correction for line in lines]
cv2.imwrite('orig.png', images[-1])

# column flipped data
images = images + [cv2.flip(image, 1) for image in images]
measurements = measurements + [-x for x in measurements]
cv2.imwrite('flip.png', images[-1])

X_train = np.array(images)
y_train = np.array(measurements)

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
plot_model(model, to_file='model.png', show_shapes=True)

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# print (SVG(model_to_dot(model).create(prog='dot', format='svg')))
# print (model_to_dot(model).create(prog='dot', format='svg').encode('utf-8'))


# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data =
#     validation_generator,
#     nb_val_samples = len(validation_samples),
#     nb_epoch=5, verbose=1)

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
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
