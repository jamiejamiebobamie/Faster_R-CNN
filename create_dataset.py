import os
import glob
from PIL import Image
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
training_labels_filepath = (dir_path
                    + '/kitti-object-detection/kitti_single/training/label_2/')
training_images_filepath = (dir_path
                    + '/kitti-object-detection/kitti_single/training/image_2/')

# the width and height of the majority of the images in the dataset
TARGET_SIZE = (1242, 375)

# the file to write to
dataset_file = open("dataset.txt","a+")

training_images = glob.glob(training_images_filepath + "*")
for file_path in training_images:
    im = Image.open(file_path)
    w, h = im.size

    image_is_target_size = (w, h) == TARGET_SIZE

    if image_is_target_size:
        file_name = file_path.split('/')[-1]
        file_name_without_ext = file_name.split(".")[0]
        label_file_path = (training_labels_filepath
                                + file_name_without_ext +'.txt')

        label_file = open(label_file_path,"r")
        for line in label_file:
            labels = line.split(" ")

            class_name = labels[0]

            if class_name == 'Pedestrian' or class_name == 'Car':
                bounding_box = labels[4:8]
                write_line = file_path+"," + ",".join(bounding_box)+","+class_name +"\n"
                print(write_line)
                dataset_file.write(write_line)
        label_file.close()
dataset_file.close()
