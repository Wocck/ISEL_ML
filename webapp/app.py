# webapp/app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from scripts.r1_model import OneRClassifier
from scripts.id3_model import predict_id3
from scripts.naive_bayes_model import predict_nb

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")

model_1r = OneRClassifier(Path("data/r1_dataset.tab"))
model_1r.fit("lenses")

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

    if model == "1r":
        result = model_1r.predict_from_values(age_group, disease_name, astigmatic_bool, tear_rate)
    elif model == "id3":
        result = predict_id3(age_group, disease_name, astigmatic_bool, tear_rate)
    else:
        result = predict_nb(age_group, disease_name, astigmatic_bool, tear_rate)

    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
