import joblib #to save the model (not like pickel)
import re 
from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional 

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

@app.get('/')

def root():
    return {"message" : "Prediction Tirage Euro Million! We gonna be rich my friend"}

@app.get("/items/{item_id}")
async def read_combi(i):
    return 1

@app.post('/api/predict')

def get_prediction():
    #Donne la proba pour que la combi d'entrée soit gagnante
    return "proba gain + proba perte"



@app.get('/api/predict')

def generate_combi():
    # Genere une combi gagnante ( avec une proba gain élevée)
    # à définir c'est quoi élevée
    return "This combi is good"


#@app.get('api/model') 

