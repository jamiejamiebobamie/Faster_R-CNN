
# import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

# from sklearn.externals import joblib
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

import os
import glob
from PIL import Image
import numpy as np

from utils import print_keras_model_layers


dir_path = os.path.dirname(os.path.realpath(__file__))
#
#
# # testing images that come with the dataset do not have labels to verify
#     # if the output labels are correct ...
# test_filepath = (dir_path
#                     + '/kitti-object-detection/kitti_single/testing/image_2/*')
# test_images = glob.glob(test_filepath)
#
#
#
# training_labels_filepath = (dir_path
#                     + '/kitti-object-detection/kitti_single/training/label_2/')
# training_images_filepath = (dir_path
#                     + '/kitti-object-detection/kitti_single/training/image_2/')
#
# training_images = glob.glob(training_images_filepath + "/*")
#
# # histogram of training/testing images based on aspect ratio:
# # training:
# # {(1242, 375): 6057, (1238, 374): 358, (1224, 370): 770, (1241, 376): 296} 4
# # testing:
# # {(1242, 375): 6579, (1224, 370): 868, (1226, 370): 71} 3
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

# training_image_filepath_to_label_filepath_lookup_dict = dict()
# for file_path in training_images:
#     im = Image.open(file_path)
#     w, h = im.size
#     image_is_target_size = (w, h) == TARGET_SIZE
#     if image_is_target_size:
#         file_path_without_ext = file_path.split(".")[0]
#         label_file_path = file_path_without_ext + '.txt'
#         training_image_filepath_to_label_filepath_lookup_dict[file_path] = (
#                                                                 label_file_path)
#
# # K-fold Cross Validation
# training_dataset, validation_dataset = 1,2
#
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
#
# # 6057 entries [0:6056]
# X = np.zeros(6057)
# y = np.zeros(6057)
#
# print(training_image_filepath_to_label_filepath_lookup_dict)
#
# # split into 67% for train and 33% for test
# X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=seed)
#
# # for img in training_image_filepath_to_label_filepath_lookup_dict.keys():
# #     read_img = cv2.imread(img, cv2.IMREAD_COLOR)
# #     X.append(read_img)
#
# set_of_test_images_at_correct_size = set()
# for file_path in test_images:
#     im = Image.open(file_path)
#     w, h = im.size
#     image_is_target_size = (w, h) == TARGET_SIZE
#     if image_is_target_size:
#         set_of_test_images_at_correct_size.add(file_path)
#

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)

img_path = (dir_path
                    + '/kitti-object-detection/kitti_single/testing/image_2/'
                    + '000000.png')
img = image.load_img(img_path, target_size=TARGET_RESIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
feature_map = model.predict(x)

input_shape = feature_map[0].shape
num_classes = 2 # 'Pedestrian' and 'Car'
epochs = 3

# building the RPN
# mirrors the architecture of the last 3 layers of VGG16
RPN_class = Sequential()
RPN_class.add(Conv2D(256, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
RPN_class.add(Conv2D(256, (1, 1), activation='relu'))
RPN_class.add(Conv2D(256, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(MaxPooling2D(pool_size=(2, 2)))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Conv2D(512, (1, 1), activation='relu'))
RPN_class.add(Dense(num_classes, activation='sigmoid'))

RPN_bb = Sequential()
RPN_bb.add(Conv2D(256, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
RPN_bb.add(Conv2D(256, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(256, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(MaxPooling2D(pool_size=(2, 2)))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Conv2D(512, (1, 1), activation='relu'))
RPN_bb.add(Dense(num_classes, activation='softmax'))

# compiling the model.
optimizer = keras.optimizers.Adadelta()
RPN_bb.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
RPN_class.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

RPN_bb.summary()

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='sigmoid'))
#
# # compiling the model.
# optimizer = keras.optimizers.Adadelta()
# model.compile(loss=keras.losses.binary_crossentropy,
#               optimizer=optimizer,
#               metrics=['accuracy'])

# # fitting the model.
# steps = len(train) // batch_size
# history = model.fit(,
#                     steps_per_epoch=steps,
#                     epochs=epochs, verbose=1)
# training_accuracy = int(history.history['acc'][0] * 100)
# print("Model's training accurcy: " + str(training_accuracy) + "%")
#
# steps = len(test) // batch_size
# validation_generator = data_gen(test, batch_size=batch_size)
#
# # there's something wrong with this method that is producing incorrect results.
# # I'm getting all images classified as containing "no_fire".
# Y_pred = model.predict_generator(validation_generator, steps, verbose=1)
# y_pred = np.argmax(Y_pred, axis=1)
# predictions = []
# for i, val in enumerate(y_pred):
#     predictions.append([val])
