# Faster R-CNN

This is a Keras implementation of Faster R-CNN. It takes large code snippets from kentaroy47's implementation of Faster R-CNN, which can be found [here](https://github.com/kentaroy47/frcnn-from-scratch-with-keras) .

## Getting Started

* Clone the repo locally:
```git clone git@github.com:jamiejamiebobamie/Faster_R-CNN.git```
* In your terminal, navigate to the main folder of the cloned repo.
* Install the requirements:
```pip install -r requirements.txt```.
* You'll need the Kitti Dataset. Download it [here](https://www.kaggle.com/twaldo/kitti-object-detection/download) .
* Place the downloaded 'kitti-object-detection' directory in the main project folder.
* Make a subdirectory:
```mkdir model_trained```
* Training takes a long time, so download the pickled model from my [google drive](www.google.com) and place the downloaded model in the 'model_trained' folder.
* If you wish to train the model yourself, simply ignore the above step.
* The built model is trained to recognize "Cars" and "Pedestrians".

Your files and folder structure should look like this:

main project folder
├── kentaroy47                      # Python code from [kentaroy47's repo](https://github.com/kentaroy47/frcnn-from-scratch-with-keras)
├── kitti-object-detection          # Downloaded dataset from [Kaggle](https://www.kaggle.com/twaldo/kitti-object-detection)
│   └── kitti_single               
│       ├── testing
│       │   └── image_2
│       └── training
│           ├── image_2
│           └── label_2
├── model_trained                   # Trained model folder.
│   └── model_frcnn.vgg.hdf5        # Pickled/built model from my [Google drive](www.google.com).
├── results_images                  # Results images.
├── utils
└── ...[files]...

## Prediction
* To make a prediction type:
```python3 main.py args```
in your terminal and press 'enter'.
'args' should be absolute filepaths of images you wish to predict.
* Prediction results are saved in results_images

### Prerequisites

The required packages are listed in the requirements file and are downloaded using the
```pip install -r requirements.txt``` command in your terminal.

## Authors

* **Jamie McCrory**
* **kentaroy47**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* https://github.com/kentaroy47/frcnn-from-scratch-with-keras
* https://github.com/broadinstitute/keras-rcnn
