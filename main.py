from train import train_model
import sys
import glob
import os


if __name__ == "__main__":
    images = []
    for arg in sys.argv[1:]:
        images.append(arg)

    currentDirectory = os.getcwd()

    files_in_model_trained_folder = glob.glob(currentDirectory
                                                + '/model_trained/*')

    folder_empty = len(files_in_model_trained_folder) == 0

    if folder_empty:
        train_model()
    else:
        model_found = (files_in_model_trained_folder[0]
                == (currentDirectory +'/model_trained/model_frcnn.vgg.hdf5'))
        if model_found:
                print('model found. \nbeginning prediction(s).')
        else:
            train_model()
