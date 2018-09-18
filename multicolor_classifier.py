#!/usr/bin/env python3
'''
Author: Ritwik Biswas and David Huang
Description: Multicolor classifier
Splits pictures into NxN pixel windows, determines color of box and stores color distribution
'''

import os
import getopt, sys
import string
from time import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle

#sklearn imports
from sklearn.svm import SVC #support vector machine
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score #accuracy scoring

#Pretty print
import colorama
from colorama import Fore, Back
from colorama import init
init(autoreset=True)

def extract_model(filename):
    '''
    Extract pickled model
    '''
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def extract_label_map(filename):
    '''
    Extract label as pickle
    '''
    label_map = {}
    with open(filename,'r') as flabel:
        for line in flabel:
            spl = line.split(",")
            label_map[int(spl[0])] = spl[1][:-1]
    return label_map

def extract_windows(image_path, window_width, window_height):
    '''
    extracts rgb values of image broken into windows of width x heigh
    '''
    import numpy as np
    from skimage.io import imread

    # read image from path
    # im = imread('dataset/COLOR/ORANGE/281620453_detail.jpg')
    im = imread(image_path)

    # set block size to 3x3
    wnd_r = window_width
    wnd_c = window_height

    rgb_list = []

    # split image into blocks and compute block average
    for r in range(0, im.shape[0] - wnd_r, wnd_r):
        col = 0
        for c in range(0, im.shape[1] - wnd_c, wnd_c):
            window = im[r:r + wnd_r, c:c + wnd_c]
            avg = np.mean(window, axis=(0, 1))
            rgb_list.append(avg)

    return rgb_list, im

def make_predictions(rgb_list, model_path, label_map_path,im):
    '''
    extract pickled model and load
    run classification for any pictures in directory
    potentially show rgb graph
    '''
    
    # extract model
    clf = extract_model(model_path)

    # extract labels
    label_map = extract_label_map(label_map_path)

    #test on single image
    prediction_list = clf.predict(rgb_list)

    #hash of color prediction
    color_store = {}
    for i in prediction_list:
        name = label_map[i]
        if name in color_store:
            color_store[name] += 1
        else:
            color_store[name] = 1
    
    total_size = len(rgb_list)
    sorted_color_store = sorted(color_store.items(), key=lambda kv: kv[1], reverse=True)

    #Determine majority colors
    percent_total = 0
    final_string = ""
    for i in sorted_color_store:
        final_string = final_string + i[0] + ", "
        temp = float(i[1]/total_size)
        percent_total += temp
        print(temp)
        if percent_total >= 0.8:
            break
    final_string = final_string[:-2]
    print(final_string)
    print(percent_total)

    f = plt.figure()
    f.suptitle(final_string, fontsize=20)
    f.add_subplot(1, 2, 1).axis("off")
    plt.imshow(im)
    f.add_subplot(1, 2, 2)
    plt.bar(list(color_store.keys()), color_store.values(), color='grey')
    plt.xticks(rotation=90)
    plt.show(block=True)

    


def prediction_driver():
    '''
    make predictions on image
    '''

    #define path and parameters
    #img_path = '/Users/ritwikbiswas/rgb-classifier/color_master/ORANGE/466178005_detail.jpg'
    #img_path = '/Users/ritwikbiswas/rgb-classifier/test_color/281620790_detail.jpg'
    img_path = '/Users/ritwikbiswas/rgb-classifier/color_master/ORANGE/511473006_detail.jpg'
    model_path = 'adaboost_0.99_0.79.sav'
    label_path = 'labels.txt'
    window_size = 3

    #extract rgb list
    rgb_list,image = extract_windows(img_path,window_size,window_size)

    #run prediction
    make_predictions(rgb_list,model_path,label_path,image)

def main():
    '''
    Driver function for extraction and training
    '''
    prediction_driver()
    


if __name__ == '__main__':
    main()
