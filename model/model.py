import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
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
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

    
# Fonction qui génère un tirage random à une date donnée
def randomNumberList(date,a,b,c,d,e,f,g):
    if (a != b != c != d != e and f != g):
        res = [date,a,b,c,d,e,f,g,0,0]
    else:
        res = randomNumberList(date,np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,12),np.random.randint(1,12))
    return res

def generateFalseTirage(df):
    for row in df.iterrows():
        # Initialisation des variables utiles
        same= 'true'
        same1= 'true'
        same2= 'true'
        same3= 'true'

        # Initialisation des tirages faux ( on vérifie qu'ils ne soit pas égaux)
        randomList = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
        randomList1 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
        randomList2 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
        randomList3 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
        while (np.array_equal(randomList, randomList1)):
            randomList1 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
        
        while (np.array_equal(randomList, randomList2) or np.array_equal(randomList1, randomList2)):
            randomList2 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)

        while (np.array_equal(randomList, randomList3) or np.array_equal(randomList1, randomList3) or np.array_equal(randomList2, randomList3)):
            randomList3 = randomNumberList(row[1]['Date'],1,1,1,1,1,1,1)
    
        
        i=1

        # On vérifie qu'ils soit différents des tirages gagnants
        for i in range(7):
            if (randomList[i+1] != row[1].to_numpy()[i+1]):
                same='false'
            if (randomList1[i+1] != row[1].to_numpy()[i+1]):
                same1='false'
            if (randomList2[i+1] != row[1].to_numpy()[i+1]):
                same2='false'
            if (randomList3[i+1] != row[1].to_numpy()[i+1]):
                same3='false'
    
        # On écrits les résultats dans le csv
        if (same=='false'):
            with open('../data/EuroMillions_numbers.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(randomList)
                writer.writerow(randomList1)
                writer.writerow(randomList2)
                writer.writerow(randomList3)
    return df

def labelisation(row):
    if row["Gain"] == 0:
        return "perdu"
    else :
        return "gagnée"
        
def preprocessor(df):
    df ["label"] = df.apply(lambda row: labelisation(row), axis = 1)
    le = preprocessing.LabelEncoder()

    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
    
    df = generateFalseTirage(df)

    return df


df = pd.read_csv("../data/EuroMillions_numbers.csv")
df = df.apply(preprocessor(df))


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

