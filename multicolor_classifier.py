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

def main():
    '''
    Driver function for extraction and training
    '''

if __name__ == '__main__':
    main()
