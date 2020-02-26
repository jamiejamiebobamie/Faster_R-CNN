import glob
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

test_filepath = dir_path + '/kitti-object-detection/kitti_single/testing/image_2/*'
test_images = glob.glob(test_filepath)

training_labels_filepath = dir_path + '/kitti-object-detection/kitti_single/training/label_2/'
training_images_filepath = dir_path + '/kitti-object-detection/kitti_single/training/image_2/'
training_images = glob.glob(training_images_filepath + "/*")

image_filepath_to_label_filepath_lookup_dict = dict()
image_size_histogram = dict()
# for file_path in training_images:
#     file_path_without_ext = file_path.split(".")[0]
#     label_file_path = file_path_without_ext + '.txt'
#     image_filepath_to_label_filepath_lookup_dict[file_path] = label_file_path
#     im = Image.open(file_path)
#     w, h = im.size
#     if (w,h) not in image_size_histogram:
#         image_size_histogram[(w,h)] = 1
#     else:
#         image_size_histogram[(w,h)] += 1

for file_path in test_images:
    # file_path_without_ext = file_path.split(".")[0]
    # label_file_path = file_path_without_ext + '.txt'
    # image_filepath_to_label_filepath_lookup_dict[file_path] = label_file_path
    im = Image.open(file_path)
    w, h = im.size
    if (w,h) not in image_size_histogram:
        image_size_histogram[(w,h)] = 1
    else:
        image_size_histogram[(w,h)] += 1
test_images

# training_labels = glob.glob(training_labels_filepath + "/*")

# labels_set = set()
# for file_path in training_labels:
#     with open(file_path, 'r') as fh:
#         for line in fh:
#             label = line.split(" ")[0]
#             labels_set.add(label)
# print(len(labels_set),labels_set)

# training:
# {(1242, 375): 6057, (1238, 374): 358, (1224, 370): 770, (1241, 376): 296} 4
print(image_size_histogram, len(image_size_histogram))
# (1242, 375): 6057 <- only using these images

# testing:
# {(1242, 375): 6579, (1224, 370): 868, (1226, 370): 71} 3

input_shape = (1242, 375, 3)
num_classes = 2 # 'Pedestrian' and 'Car'
batch_size = 7
epochs = 3

# building the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

# compiling the model.
optimizer = keras.optimizers.Adadelta()
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

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
