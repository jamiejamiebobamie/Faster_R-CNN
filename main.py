from train import train_model
from predict import predict
import sys
import glob
import os


if __name__ == "__main__":
    images = []
    for arg in sys.argv[1:]:
        images.append(arg)
    print(len(images))

    currentDirectory = os.getcwd()

    files_in_model_trained_folder = glob.glob(currentDirectory
                                                + '/model_trained/*')

    folder_empty = len(files_in_model_trained_folder) == 0

    if folder_empty:
        train_model()
    else:
        model_not_found = (files_in_model_trained_folder[0]
                != (currentDirectory +'/model_trained/model_frcnn.vgg.hdf5'))
        if model_not_found:
            print('model not found. check for \'model_frcnn.vgg.hdf5\''+
                'in \'model_trained\' folder.')
            print('training model.')
            train_model()
        else:
            print('model found.')
    print('Begin prediction(s).')

    no_args = len(images) == 0
    if no_args:
        static_file_path = (currentDirectory
                            + '/kitti-object-detection/kitti_single/testing'
                            +'/image_2/000007.png')
        predict(static_file_path)
    else:
        for image in images:
            predict(static_file_path)
