import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from os import SCHED_OTHER
import random
import csv
from preprocessing import *


#unzipping the dataset csv

import zipfile
with zipfile.ZipFile("../data/dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("../data")



h = 0.02  # step size in the mesh

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


df = pd.read_csv("../data/dataset/EuroMillions_numbers.csv", sep=";", date_parser=True)
df = df.apply(preprocessor(df))

def labelisation(row):
    if row["Gain"] == 0:
        return "perdu"
    else :
        return "win"
    
for row in df : df ["label"] = df.apply(lambda row: labelisation(row), axis = 1)

#train-test split 
X = df.drop('label', axis = 1)
y = df['label']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Train the dataset of various models written above and choose the more accurate


#Load the model using joblib 

