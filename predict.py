from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
from kentaroy47.config import Config as config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import kentaroy47.roi_helpers as roi_helpers
import argparse
import os
import kentaroy47.resnet as nn
from kentaroy47.visualize import draw_boxes_and_label_on_image_cv2
from kentaroy47.predict import predict_single_image, parse_args



def predict(args_):

    # from: https://github.com/kentaroy47/frcnn-from-scratch-with-keras

    path = args_
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier_only, cfg, class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping)
