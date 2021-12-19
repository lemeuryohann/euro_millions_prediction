# euro_millions_prediction
Predict model with FastAPI and panda.


## App's architecture 
| api
    | main.py
| data 
    | EuroMillions_numbers.csv
| model
    | model.py
    | preprocessing.py
| README.md
| requirements.txt 


## How to deploy 
 - Clone our git repo (more details below)
 - create or use your virtual environment set on python3
 - install the requirements.txt file
 - then on your terminal start your fast api app : uvicorn main:app --reload (make sure you are working in the right directory:api )
 - on your local host : http://127.0.0.1:8000/docs you should be able to view all our app features
 - 