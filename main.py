from train import train_model
import sys
import os

if __name__ == "__main__":
    images = []
    for arg in sys.argv[1:]:
        images.append(arg)

# https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
    number_of_files_in_model_trained_folder = len(
                                        [name for name in
                                            os.listdir('./model_trained')
                                            if os.path.isfile(name)]
                                        )
    if number_of_files_in_model_trained_folder > 0:
        if files_in_model_trained_folder[0] == 'model_frcnn.vgg.hdf5':
            print('yep')
        else:
            print('nope')
            train_model()
