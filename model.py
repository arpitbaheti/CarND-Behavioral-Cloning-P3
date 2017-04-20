import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn


MAX_EPOCHS = 30
log_file = '/home/arpitb/2tb/home/arpitb/simulator/6/driving_log.csv'

def Model():

    keep_prob = 0.2
    ch, row, col = 3, 66, 200

    model = Sequential()

    '''
    Lamda layer for Normalization
    Input Shape = 66 x 200 x 3
    '''
    model.add(Lambda (lambda x: x / 255 - 0.5,
                    input_shape=(row, col, ch),
                    output_shape=(row, col, ch)))

    '''
    First Convolution layer with Depth = 24
    kernel size  = 5 x 5
    stride = 2 x 2
    activation = tanh
    '''
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))

    '''
    Second Convolution layer with Depth = 36
    kernel size  = 5 x 5
    stride = 2 x 2
    activation = tanh
    '''
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))

    '''
    Third Convolution layer with Depth = 48
    kernel size  = 5 x 5
    stride = 2 x 2
    activation = tanh
    '''
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

    '''
    Forth Convolution layer with Depth = 64
    kernel size  = 3 x 3
    stride = 1 x 1
    activation = tanh
    '''
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    '''
    Fifth Convolution layer with Depth = 64
    kernel size  = 3 x 3
    stride = 1 x 1
    activation = tanh
    '''
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    '''
    Flatten the image
    '''
    model.add(Flatten())

    '''
    First Dense layer with output size = 100
    '''
    model.add(Dense(100, activation='relu'))

    '''
    First Dropout Layer
    '''
    model.add(Dropout(keep_prob))

    '''
    Second Dense layer with output size = 50
    '''
    model.add(Dense(50, activation='relu'))

    '''
    Second Dropout Layer
    '''
    model.add(Dropout(keep_prob))

    '''
    Third Dense layer with output size = 10
    '''
    model.add(Dense(10, activation='relu'))

    '''
    Final Logit Layer
    '''
    model.add(Dense(1))

    return model

samples = []
with open(log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def crop_image(image):

    image = image[60: 135, 0: image.shape[1]]

    return image

def preprocess(image):

    # Crop the image
    image = crop_image(image)

    # Augment brightness
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()

    # scaling up or down the V channel of HSV
    image[:,:,2] = image[:,:,2] * random_bright

    # reshape the image
    image = cv2.resize(image, (200,66))

    return image




def generator(samples, batch_size = 32):
    correction = 0.2
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:

                for cam_view in range(3):

                    name = batch_sample[cam_view]

                    if os.path.exists(name):
                        image = cv2.imread(name)
                        angle = float(batch_sample[3])

                        if cam_view == 1:
                            angle = angle + correction
                        elif cam_view == 2:
                            angle = angle - correction

                        # Flip the image
                        flip_image = cv2.flip(image, 1)
                        flip_angle = -angle

                        # Preprocess the image
                        image = preprocess(image)
                        flip_image = preprocess(flip_image)

                        # append into database
                        images.append(np.reshape(image, (66,200,3)))
                        images.append(np.reshape(flip_image, (66,200,3)))

                        angles.append(angle)
                        angles.append(flip_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def valid_generator(samples, batch_size = 32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:

                name = batch_sample[0]

                if os.path.exists(name):
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])

                    image = crop_image(image)

                    # reshape the image
                    image = cv2.resize(image, (200,66))

                    # change colourspace
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                    # append into database
                    images.append(np.reshape(image, (66,200,3)))
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = valid_generator(validation_samples, batch_size = 128)

model = Model()

adam = Adam(lr = 0.0001)

model.compile(loss = 'mse', optimizer = adam)

print("Model summary:\n", model.summary())


# Early syopping if loss on val doesn't decreases in 3 continues epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3,
                           verbose=0, mode='min')

'''
 fit_generator with sample_per_epoch = len(train_samples)*6 as we are using
 flipped images and all three camera images in training.
'''
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6,
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=5)

model.save('model.h5')
