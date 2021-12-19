from os import SCHED_OTHER
import random
import csv
import numpy as np 
from sklearn import preprocessing 

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
            with open('../EuroMillions_numbersNEW.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(randomList)
                writer.writerow(randomList1)
                writer.writerow(randomList2)
                writer.writerow(randomList3)
    return df

def encoding(df):
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
    return df 
def preprocessor(df):

    df = df.apply(encoding(df))
    df = generateFalseTirage(df)

    return df