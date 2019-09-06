#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:03:16 2019

@author: nuoxu
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from east.label import point_inside_of_quad
from east.network import East
from east.preprocess import resize_image
from east.nms import nms

from east.predict import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/jz22.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    # Specift a GPU
    os.environ["CUDA_VISIBLE_DEVICES"]='5'
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    
    saved_model_weights_file_path = 'saved_model/east_model_weights_3T736.h5'
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)