# import joblib #to save the model (not like pickel)
import re 
from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional
import csv
import joblib

class combinaison(BaseModel):
    combi_id : str
    description : Optional[str] = None
    N1 : int 
    N2 : int 
    N3 : int
    N4 : int
    N5 : int
    E1 : int
    E2 : int
    probToWin : Optional[int] = None 
    
app = FastAPI()

model = joblib.load('chosen_model.joblib')

def prediction_combi(model, combi):
    label = model.predict([combi])[0]
    combi_proba = model.predict_proba([combi])


@app.get('/')

def root():
    return {"message" : "Prediction Tirage Euro Million! We gonna be rich my friend"}

@app.get("/combinaisons/{combi_id}")
async def read_combi(combi_id: int):
    return {"combinaison_id" : combi_id}

@app.post('/api/predict')

def get_prediction():
    #Donne la proba pour que la combi d'entrée soit gagnante
    return "proba gain + proba perte"



@app.get('/api/predict')

def generate_combi():
    # Genere une combi gagnante ( avec une proba gain élevée)
    # à définir c'est quoi élevée
    return "This combi is good"


@app.get('/api/model')

def get_modele():
    # Donne les infos sur le modèle ( métriques, nom de l'algo, param d'entrainement)
    return "modèle"

@app.put('/api/model')

# data doit être de la forme :
# {
#   Date : date,
#   N1 : int,
#   N2 : int,
#   N3 : int,
#   N4 : int,
#   N5 : int,
#   E1 : int,
#   E2 : int,
#   gagnant: int,
#   Gain : int
# }

def add_data(data):
    tirage = [data.Date, data.N1, data.N2, data.N3, data.N4, data.N5, data.E1, data.E2, data.gagnant, data.gain]
    # Ecriture dans le fichier des donnés passé en paramètre 
    with open('../data/EuroMillions_numbers.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(tirage)
    return "donnée ajoutée"

@app.post('/api/model/retrain')

def retrain_modele():
    # réentrainer le modèle avec les nouvelles données
    return "modèle réentrainer"


