from pyexpat import model
from nbformat import write
import streamlit as st
import tensorflow as tf
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


def predict(instances):

    payload = {"instances": [instances]}

    res = requests.post(
        url="http://localhost:8605/v1/models/dry_bean_model:predict", json=payload
    )

    preds = res.json()
    return preds


idx2class = {
    0: "BARBUNYA",
    1: "BOMBAY",
    2: "CALI",
    3: "DERMASON",
    4: "HOROZ",
    5: "SEKER",
    6: "SIRA",
}

st.sidebar.markdown(
    "<h1><center>Dry bean classifier</center></h1>", unsafe_allow_html=True
)

res = st.sidebar.selectbox(
    "Choose Model", options=["Random Forest", "SVM", "Vanila NN"]
)

st.sidebar.markdown(
    f"<h2>Classes</h2>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"<ul>{''.join(['<li>{}</li>'.format(label) for label in idx2class.values()])}</ul>",
    unsafe_allow_html=True,
)

st.markdown("<h3><center>Upload test.csv file</center></h3>", unsafe_allow_html=True)

file_uploader = st.file_uploader("")

if file_uploader is not None:
    df = pd.read_csv(file_uploader)

    st.dataframe(data=df.head())

model = None

if res == "Random Forest":
    model = joblib.load("ML_models\\RandomForest-tuned.joblib")

if res == "SVM":
    model = joblib.load("ML_models\\SVM(SGD-Hinge)-tuned.joblib")

if res == "Vanila NN":
    st.warning("Using the served TF model version 2....")

    X = df.values[:5, :]
    X = (X - X.mean()) / X.std()

    with st.spinner("Classifying...."):
        preds = predict(X.tolist())

if model:

    btn = st.button("predict")

    if res == "Random Forest":
        X = df.values[:5, :]

    elif res == "SVM":
        X = df.values[:5, :]
        X = (X - X.mean()) / X.std()

    if btn:
        with st.spinner("Classifying...."):
            preds = model.predict(X)

            for idx, pred in enumerate(preds):
                st.write(
                    f"The predicted class for instance is: {(pred, idx2class[pred])}"
                )
