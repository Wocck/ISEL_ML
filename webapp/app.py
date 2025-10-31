# webapp/app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.r1_model import OneRClassifier
from src.id3_model import ID3Classifier
from src.naive_bayes_model import NaiveBayesClassifier
from src.config import TAB_DATASET_FILE

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")

df = pd.read_csv(TAB_DATASET_FILE, sep="\t")

model_1r = OneRClassifier()
model_1r.set_training_data(df)
model_1r.fit("lenses")

model_id3 = ID3Classifier()
model_id3.set_training_data(df)
model_id3.fit("lenses")

model_naive_bayes = NaiveBayesClassifier()
model_naive_bayes.set_training_data(df)
model_naive_bayes.fit("lenses")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    model: str = Form(...),
    age_group: str = Form(...),
    disease_name: str = Form(...),
    astigmatic: str = Form(...),
    tear_rate: str = Form(...)
):
    astigmatic_bool = (astigmatic.lower() == "yes")
    row = {
        "age_group": age_group.strip(),
        "disease_name": disease_name.strip(),
        "astigmatic": "yes" if astigmatic_bool else "no",
        "tear_rate": tear_rate.strip()
    }
    
    if model == "1r":
        result = model_1r.predict_row(row)
    elif model == "id3":
        result = model_id3.predict_row(row)
    else:
        result = model_naive_bayes.predict_row(row)

    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
