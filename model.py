import csv
import os
import numpy as np
import random
import cv2
from sklearn.utils import shuffle
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import backend
from keras.callbacks import ModelCheckpoint, EarlyStopping

cwd = os.getcwd()
samples = []

with open('sim_data_10_lap/driving_log_balanced.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Sample generator. Setup as used to produce the best performing model.
# Left and right image use was commented out due to poor validation loss.

def generator(samples, batch_size=100):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:

                if True:
                    center_name = 'sim_data_10_lap/IMG/'+batch_sample[0].split('/')[-1]
                    #left_name = 'sim_data_10_lap/IMG/'+batch_sample[1].split('/')[-1]
                    #right_name = 'sim_data_10_lap/IMG/'+batch_sample[2].split('/')[-1]

                    center_image = mpimg.imread(center_name)
                    #left_image = mpimg.imread(left_name)
                    #right_image = mpimg.imread(right_name)

                    center_image_flipped = np.fliplr(center_image)
                    #left_image_flipped = np.fliplr(left_image)
                    #right_image_flipped = np.fliplr(right_image)

                    center_angle = float(batch_sample[3])
                    #left_angle = min(float(batch_sample[3]) + offset, 1)
                    #right_angle = max(float(batch_sample[3]) - offset, -1)

                    center_angle_flipped = -center_angle
                    #left_angle_flipped = -left_angle
                    #right_angle_flipped = -right_angle
                    
                    images.append(center_image)
                    angles.append(center_angle)
                    #images.append(left_image)
                    #angles.append(left_angle)
                    #images.append(right_image)
                    #angles.append(right_angle)

                    images.append(center_image_flipped)
                    angles.append(center_angle_flipped)
                    #images.append(left_image_flipped)
                    #angles.append(left_angle_flipped)
                    #images.append(right_image_flipped)
                    #angles.append(right_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=2)
validation_generator = generator(validation_samples, batch_size=2)

# Modified NVIDIA end-to-end convolutional neural network. Images are normalized 
# and cropped in the model so nothing has to be done in the drive.py file.
def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,30), (0,0))))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='elu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='elu'))

    #Fully Connected Layer
    model.add(Flatten())
    #model.add(Dense(1164,activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50,activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10,activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1))

    adam = Adam(learn_rate)
    model.compile(loss='mse', optimizer=adam)
    return model

# Nested for loops allow iterating through multiple hyperparameters.

epochs = 10000

offset = 0.00

for batch_size in [512]:
    for learn_rate in [0.00005]:
        for drop_out in [0.00]:

            model = create_model()
            # Save model file with a detailed filename.
            model_filename = (cwd + '/models/' +
                              'bs' + str(batch_size) +
                              '-lr' + str(learn_rate) +               
                              '-do' + str(drop_out) +        
                              '-epoch' + '{epoch:03d}' +                
                              '-loss' + '{val_loss:.6f}' +   
                              '-os' + str(offset) +           
                              '-model.h5')
            # callback checks if validation loss is better than last epoch,
            # if so it saves the model.
            callback = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            # early_stop allows training to be killed if epochs > patience.
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5000, verbose=0, mode='auto')

            model.fit_generator(train_generator, 
                                steps_per_epoch=len(train_samples)/batch_size, 
                                epochs=epochs, 
                                verbose=1, 
                                callbacks=[callback, early_stop], 
                                validation_data=validation_generator, 
                                validation_steps=len(validation_samples)/batch_size)

             
            # Delete old models between epochs. Change 4 to 5 if learn_rate
            # is five sigfigs or greater. (It's displayed with an extra '-')
            for each in (os.listdir(cwd + '/models/')):
                temp = each
                temp = temp.split('-')[5]
                temp = float(temp.split('loss')[1])
                if temp > 0.005:
                    os.remove('models/' + each)
                
            



