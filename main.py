from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the model
model = joblib.load("model1.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  feature1: float = Form(...),
                  feature2: float = Form(...),
                  feature3: float = Form(...)):
    # You may need to modify this list based on how many features your model needs
    data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(data)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction[0]
    })
