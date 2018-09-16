#!/usr/bin/env python3
'''
Author: Ritwik Biswas
Description: Herman Miller Color Classifier 
Analyzes all the pictures in folders and extracts mean RGB value as a feature vector
Training/testing with:
        - SVM classifier with an rbf kernel
        - Adaboost classifier with a Decision Tree base model
'''
import os
import getopt, sys
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score #accuracy scoring

#Pretty print
import colorama
from colorama import Fore, Back
from colorama import init
init(autoreset=True)

def color_histogram():
    '''
    draw rgb histogram for a given image
    '''
    # draw histogram in python. 
    # img = cv2.imread('/Users/ritwikbiswas/herman-miller-capstone/deeplearning/test_colors/red/403901046_detail.jpg')
    img = cv2.imread('/Users/ritwikbiswas/herman-miller-capstone/deeplearning/test_colors/green/459830010_detail.jpg')

    h = np.zeros((300,256,3))
    
    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)
    
    h=np.flipud(h)
    
    cv2.imshow('colorhist',h)
    print(h)
    cv2.waitKey(0)
 
def color_mean_extractor(image_location):
    ''' 
    extract mean rgb array and returns in the form [R,G,B]
    '''
    img = cv2.imread(image_location)
    color_mean = cv2.mean(img)
    rgb_color_mean = [color_mean[2], color_mean[1], color_mean[0]]
    return rgb_color_mean

def iterate_over_images(root_dir):
    '''
    Extract images from all pictures and analyze RGB in sub folders of classes
    Return feature np array and label np array
        - feature array: [[r_1,g_1,b_1],[r_2,g_2,b_2],...[r_n,g_n,b_n]]
        - label array: [[1,1,1,1,2,2,2,2,......,n,n,n]]
    '''

    #A direct map of index to label
    class_label_map = {}

    #Initialize feature vector and label vector
    rgb_features_train = []
    class_labels_train = []

    rgb_features_test = []
    class_labels_test = []

    #get full list of colors in path
    print(Fore.GREEN + "Extracting classes from data.")
    listdir = os.listdir(root_dir)
    print(Fore.GREEN + "Identified " + str(len(listdir)) + " classes.")

    #Loop over directory to get RGB Vectors
    label_class = 1
    total_count = 0
    #iterate through outer folder
    for directory in listdir:
        class_label_map[label_class] = str(directory)

        count = 1 
        print("Extracting data from " + directory)

        #Iterate through each class directory
        full_path = str(root_dir + directory + "/")
        list_subdir = os.listdir(full_path)
        for pic in list_subdir:
            # if count == 301: #how much data we want for testing (remove for all data)
            #     break

            #extract mean color for a given picture
            rgb_mean = color_mean_extractor(full_path+pic)
            total_count += 1

            #add mean and labels to vectors
            if count%100 == 0: # 1% of data is split for testing
                rgb_features_test.append(rgb_mean)
                class_labels_test.append(label_class)
            else:
                rgb_features_train.append(rgb_mean)
                class_labels_train.append(label_class)
            count += 1

        label_class += 1
    
    #convert data to np array
    features_train = np.array(rgb_features_train)
    labels_train = np.array(class_labels_train)

    features_test = np.array(rgb_features_test)
    labels_test = np.array(class_labels_test)

    print(Fore.MAGENTA + "Extracted " + str(total_count) + " images for training and testing.")
    return features_train, labels_train, features_test, labels_test, class_label_map
    
def graph_data(features, labels, label_map):
    '''
    graph RGB feature vectors in 3D space with appropriate classes marked
    '''
    # print(features)
    # print(labels)
    # print(label_map)
    fig = plt.figure()
    ax = Axes3D(fig)

    #initialize empty variables
    x_r = []
    y_g = []
    z_b = []
    color = []
    color_map = {"BEIGE" : "#ffcc66", "BLACK" : "#000000", "BLUE" : "#0000ff", "BROWN" : "#996633", "GRAY" : "#a7a7a7", "GREEN" : "#009933", "METALLIC" : "#dbd9d9", "MISC" : "#ccffcc", "ORANGE" : "#ff6600", "PINK" : "#ff66ff", "RED" : "#ff0000", "VIOLET" : "#993399", "WHITE" : "#ffffcc", "YELLOW" : "#e8f106"}
    #iterate through np array of features to assign to r,g,b --> x,y,z
    for i in range(0,len(features)):
        # print(features[i][0])
        # print(features[i][1])
        # print(features[i][2])
        x_r.append(features[i][0])
        y_g.append(features[i][1])
        z_b.append(features[i][2])
        color.append(color_map[label_map[labels[i]]])
    ax.scatter(x_r, y_g, z_b, c=color, )
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()

def train_model(features_train, labels_train, features_test, labels_test, label_map):
    '''
    Train model using features and labels and pickles output model
    Support Vector Machine classifier with RBF Kernal
    Adaboost with Decision Tree as base classifier
    SVC options here
     SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
         max_iter=-1, probability=False, random_state=None, shrinking=True,
         tol=0.001, verbose=False)
    '''

    classifier_name = "adaboost"
    
    ######################### Classifier Choosing Here ########################
    #SVM Test
    #clf = SVC(kernel="rbf",gamma='auto', C=1.0) #.95/.44
    
    # Adaboost Test
    base_model_stack = tree.DecisionTreeClassifier(min_samples_split=20)
    #base_model_stack = GradientBoostingClassifier()
    clf = AdaBoostClassifier(n_estimators=100000, base_estimator=base_model_stack)
    #clf = AdaBoostClassifier(n_estimators=100000, base_estimator=base_model_stack) # Score: 0.936

    ###########################################################################

    #Training for model fit
    print (Fore.CYAN + "Begin training ...")
    t0 = time()
    clf.fit(features_train,labels_train)
    t1 = time()
    print (Fore.CYAN + "Finished training in " + str(t1-t0) + " seconds.")

    #Model Accuracy determined here
    print (Fore.GREEN + "--------------------")
    train_score = str(clf.score(features_train,labels_train))
    test_score = str(clf.score(features_test,labels_test))
    print (Fore.GREEN + "Model Train Accuracy: " + train_score[:4])
    print (Fore.GREEN + "Model Test Accuracy: " + test_score[:4])
    print (Fore.GREEN + "--------------------")

    #pickle the model for later use
    model_file = classifier_name + "_" + train_score[:4] + "_" + test_score[:4] + ".sav"
    pickle.dump(clf, open(model_file, 'wb'))
    
    #write label map to file to be extracted later
    labels_file = "labels.txt"
    with open(labels_file, 'w') as out:
        for i in label_map:
            out.write(str(i) + "," + str(label_map[i]))
            out.write("\n")
    # print(len(features_train))
    # print(len(labels_train))
    # print(len(features_test))
    # print(len(labels_test))

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

def test_model(image_directory, model_path, label_map_path):
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
    tests = []
    img_path = "/Users/ritwikbiswas/rgb-classifier/color_master/ORANGE/281620560_detail.jpg"
    rgb_mean = color_mean_extractor(img_path)
    tests.append(rgb_mean)
    print(clf.predict(tests))


    #Eventually write tests on full image directory
    # loop through director, add classifications to vector, loop through both img vectors and classifications to display classification/image

def main():
    '''
    Driver function for extraction and training
    '''

    # Define root directory to training data <<CHANGE HERE>>
    root_dir = '/Users/ritwikbiswas/rgb-classifier/color_master/'

    #Extract RGB Features
    features_train, labels_train, features_test, labels_test, label_map = iterate_over_images(root_dir)

    #Graph Data on 3D axis to check for clustering
    #graph_data(features, labels, label_map)

    #train model here
    train_model(features_train, labels_train, features_test, labels_test, label_map)

    # Define test directory with pictures to train on
    #test_dir = '/Users/ritwikbiswas/rgb-classifier/test_color'

    #test model here
    #model_path = '/Users/ritwikbiswas/rgb-classifier/svm_0.76_0.63.sav'
    #label_map_path = '/Users/ritwikbiswas/rgb-classifier/labels.txt'
    #test_model(test_dir, model_path, label_map_path)

if __name__ == '__main__':
    main()
