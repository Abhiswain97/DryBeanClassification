import tensorflow as tf
import requests
import pandas as pd
import joblib


def predict(instances):

    payload = {"instances": [instances]}

    res = requests.post(
        url="http://localhost:8501/v1/models/saved_model:predict",
        json=payload,
    )

    preds = res.json()
    return preds


if __name__ == "__main__":

    idx2class = {
        0: "BARBUNYA",
        1: "BOMBAY",
        2: "CALI",
        3: "DERMASON",
        4: "HOROZ",
        5: "SEKER",
        6: "SIRA",
    }

    df = pd.read_csv("full_feats_test.csv")
    X = df.values

    X_test = X[:5, :]
    scaler = joblib.load("../ML_models/NN_scaler.scaler")

    X_test_scaled = scaler.transform(X_test)

    predictions = []

    for i, ins in enumerate(X_test_scaled):
        pred = predict(instances=ins.tolist())
        pred = pred["predictions"]
        idx = tf.argmax(pred, axis=1)

        predictions.append(idx2class[idx.numpy()[0]])

    print(predictions)
