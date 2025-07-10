from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()

# Загружаем модель, сохранённую train.py
model = load("model.joblib")

class IrisIn(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(item: IrisIn):
    arr = np.array(item.data).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": int(pred)}