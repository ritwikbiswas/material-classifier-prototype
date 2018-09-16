
8084 total images in datasets split in 15 color classes
Classifier Accuracy Information (99% training, 1% testing)

svm (rbf kernel) -----> train:95%, test:43%
adaboost (n-est = 1) [decision tree(split = 15)] -----> train:95%, test:43%
adaboost (n-est = 10) [decision tree(split = 15)] -----> train:79%, test:65%
adaboost (n-est = 100) [decision tree(split = 15)] -----> train:99%, test:75%
adaboost (n-est = 100) [decision tree(split = 20)] -----> train:99%, test:80%
adaboost (n-est = 1000) [decision tree(split = 20)] -----> train:99%, test:80%
adaboost (n-est = 1000) [decision tree(split = 15)] -----> train:99%, test:80%
adaboost (n-est = 1000) [gradient boost] -----> train:92%, test:69%

adaboost (n-est = 1000) [decision tree(split = 20)] -----> train:%, test:%
