import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import random
import csv
#from generateLosingCombi import *

#list of models to train on our dataset and then choose the more accurate

names = [
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
]

classifiers = [
    GaussianProcessClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
]


df = pd.read_csv("../data/EuroMillions_numbersComplete.csv", sep=",")

#labelisation des données : combi perdante ou gagnante 
def labelisation(row):
    if row["Gain"] == 0:
        return "perdu"
    else :
        return "win"
    
df ["label"] = df.apply(lambda row: labelisation(row), axis = 1)
print(df.head())
print(df.dtypes)

le = preprocessing.LabelEncoder()

for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


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


from sklearn.metrics import classification_report, accuracy_score
for name, classifier in zip(names, classifiers):
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(name, score)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test,y_pred))
    print('Accuracy: {} %'.format(100*accuracy_score(y_test, y_pred)))

chosenClf = GaussianProcessClassifier()
chosenClf.fit(X_train, y_train)

#Load the model using pickle 

# We found that the gaussian process is the most accurate and has more precision ! 
# Let's load the gaussian classifier with pickle 

import pickle
filename= "saved_model.sav"
pickle.dump(chosenClf, open(filename, 'wb'))



