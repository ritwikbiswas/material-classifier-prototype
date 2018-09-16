#!/usr/bin/env python3
'''
Author: Ritwik Biswas
Description: Testing RGB classifier trained for classifying Herman Miller data
'''

import os
import sys
import string
from time import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle

#sklearn imports
from sklearn.svm import SVC #support vector machine
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
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

def main():
    '''
    Driver function for testing
    '''
    model_file_name = "to be continued"
    extract_model(model_file_name)
    
if __name__ == '__main__':
    main()