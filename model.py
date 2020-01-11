import numpy as np
import cv2
import pandas as pd
import random

def format_img_path(path, folder='images'):
    fmt_path = path.replace(r'C:\Users\Vijay Yalamanchili\Desktop\GIT_Repos\Udacity_BehaviorCloning\{}'.format(folder), folder)
    return fmt_path.replace("\\", "/")

def add_imgs_steer_folder(folder, image_Data, steer_data, prob=1.,
                          add_lr=False, steer_filt_gr=0.0, steer_filt_lt=0.0):
    with open('{}/driving_log.csv'.format(folder), 'r') as f:
        csv_data = f.readlines()

    for line in csv_data:
        # Skip some data if prob of keeping data is specified (default is 1.0)
        if (random.uniform(0, 1) > prob):
            continue
        
        # Update image paths from local to relative
        line_data = line.split(',')
        img_path = format_img_path(line_data[0], folder)

        # Load image using CV2
        img_data_tmp = cv2.imread(img_path)
        steer_tmp = float(line_data[3])
        
        if steer_filt_gr > 0. and abs(steer_tmp) < steer_filt_gr:
            continue # when steer filt gr is specified and steer value lt filt.
            
        if steer_filt_lt > 0. and abs(steer_tmp) > steer_filt_lt:
            continue # when steer filt lt is specified and steer value gr filt.
        
        image_data.append(img_data_tmp)
        steer_data.append(steer_tmp)
    #     steer_data.append([float(line_data[3]), float(line_data[4])])
            
        # ADD FLIPPED IMAGE
        image_data.append(cv2.flip(img_data_tmp, flipCode=0))
        steer_data.append(-1.*steer_tmp)
        
        # ADD LEFT, RIGHT IMAGE
        if add_lr:
#             if (random.uniform(0, 1) > 0.5):
#                 continue
                
            img_path_left = format_img_path(line_data[1], folder)
            img_path_right = format_img_path(line_data[2], folder)

            image_data.append(cv2.imread(img_path_left))
            steer_data.append(steer_tmp + 0.2)
            image_data.append(cv2.imread(img_path_right))
            steer_data.append(steer_tmp - 0.2)
    return

def rolling_mean(data, window=3):
    return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values

def generator(X_data, y_data, batch_size=32):
    num_samples = len(y_data)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_batch = X_data[offset:offset+batch_size]
            y_batch = y_data[offset:offset+batch_size]
            yield X_batch, y_batch

image_data = []
steer_data = []

# Add required training data sets
add_imgs_steer_folder('regular_driving', image_data, steer_data, add_lr=True)
add_imgs_steer_folder('recovery_driving', image_data, steer_data, add_lr=True, steer_filt_gr=0.0)
add_imgs_steer_folder('reverse_driving', image_data, steer_data, add_lr=True)

X = np.array(image_data)
y = np.array(steer_data)

seed = 1234
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=seed)

del image_data
del steer_data

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from math import ceil

# Build Convolutional Neural Network in Keras Here
dropout = 0.3
model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# # Use LeNet Architecture to start with
model.add(Conv2D(24, kernel_size=(5, 5),
                 padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=dropout))
model.add(Conv2D(36, kernel_size=(5, 5),
                 padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=dropout))
model.add(Conv2D(48, kernel_size=(3, 3),
                 padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=dropout))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(rate=dropout))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=dropout))
model.add(Dense(1))

# COMPILE MODEL
learn_rate = '0.001'
batch_size = 128
n_epochs = 20

model.compile(loss='mse', optimizer=Adam(lr=float(learn_rate)))

train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_valid, y_valid, batch_size=batch_size)

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=ceil(len(y_train)/batch_size), 
                                     validation_data=validation_generator,
                                     validation_steps=ceil(len(y_valid)/batch_size),
                                     epochs=n_epochs, verbose=1)

model_name = 'model_LR_LeNetCnn3Dense3_Drp{}_LR{}_rnd{}'.format(dropout, learn_rate, seed)
model.save(model_name + '.h5')

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(model_name + '.png')
