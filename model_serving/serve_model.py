from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from typing import List

# initiate API
app = FastAPI()

# define model for post request.
class ModelParams(BaseModel):
    params: List[List]


@app.post("/predict")
def predict(params: ModelParams):
    model = load("../ML_models/Auto_tuned_LightGBM_without_trans.model")

    preds = model.predict(params.params)
    probs = model.predict_proba(params.params)

    return {"preds": preds.tolist(), "probs": probs.tolist()}
