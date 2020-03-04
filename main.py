
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split

import os
import glob
from PIL import Image
import numpy as np

from utils import print_keras_model_layers

dir_path = os.path.dirname(os.path.realpath(__file__))

TARGET_SIZE = (1242, 375)
TARGET_RESIZE = (224, 224)

# the ratio to multiply the bounding box coordinates by to get the
# new bounding box coordinates of the scaled image.
wTARGET_RESIZE,hTARGET_RESIZE = TARGET_RESIZE
wTARGET_SIZE,hTARGET_SIZE = TARGET_SIZE

wRATIO_BB=wTARGET_RESIZE/wTARGET_SIZE
hRATIO_BB=hTARGET_RESIZE/hTARGET_SIZE
RATIO_BB = (wRATIO_BB,hRATIO_BB)
print(RATIO_BB)

# https://arxiv.org/abs/1409.1556
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)

img_path = (dir_path
                    + '/kitti-object-detection/kitti_single/testing/image_2/'
                    + '000001.png')
img = image.load_img(img_path, target_size=TARGET_RESIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
test_feature_map = model.predict(x)

print(test_feature_map)

input_shape = test_feature_map[0].shape
num_classes = 3 # 'Pedestrian', 'Car', 'Background'
epochs = 3

ANCHOR_BOX_SCALES = [128, 256, 512]
ANCHOR_ASPECT_RATIOS = [(1,1), (1,2), (2,1)]

# building the RPN
# mirrors the architecture of the last 3 layers of VGG16
# classification model
RPN_class = Sequential()
# input layer
RPN_class.add(Conv2D(256, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
#block CNN 1
RPN_class.add(Conv2D(256, (1, 1), activation='relu'))
RPN_class.add(Conv2D(256, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
#block CNN 2
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
# block CNN 3
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
# block CNN 4
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
# FC
RPN_class.add(Dense(512, activation='relu'))
RPN_class.add(Dense(512, activation='relu'))
RPN_class.add(Dense(512, activation='relu'))
#output layer
RPN_class.add(Dense(num_classes, activation='softmax'))

# bounding box RPN model
RPN_bb = Sequential()
# input layer
RPN_bb.add(Conv2D(256, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
#block CNN 1
RPN_bb.add(Conv2D(256, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(256, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
#block CNN 2
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
#block CNN 3
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
#block CNN 4
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
# FC
RPN_bb.add(Dense(512, activation='relu'))
RPN_bb.add(Dense(512, activation='relu'))
RPN_bb.add(Dense(512, activation='relu'))
# output layer
RPN_bb.add(Dense(4, activation='sigmoid')) # 4 numbers for x1,y1,x2,y2 ??

# compiling the RPN models.
optimizer = keras.optimizers.Adadelta()
RPN_bb.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])
RPN_class.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

RPN_class.summary()
RPN_bb.summary()

# example of how to freeze a layer in keras
# frozen_layer = Dense(32, trainable=False)
