
from os import SCHED_OTHER
import pandas as pd
import numpy as np
import random
import csv


#unzipping the dataset csv

import zipfile
with zipfile.ZipFile("../data/dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("../data")


# lecture du csv
df  = pd.read_csv('data/dataset/EuroMillions_numbers.csv', sep=';')


# Fonction qui génère un tirage random à une date donnée
def randomNumberList(date,a,b,c,d,e,f,g):
    if (a != b != c != d != e and f != g):
        res = [date,a,b,c,d,e,f,g,0,0]
    else:
        res = randomNumberList(date,np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,50),np.random.randint(1,12),np.random.randint(1,12))
    return res


def generateFalseTirage():
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
            with open('EuroMillions_numbers.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(randomList)
                writer.writerow(randomList1)
                writer.writerow(randomList2)
                writer.writerow(randomList3)

        

generateFalseTirage()