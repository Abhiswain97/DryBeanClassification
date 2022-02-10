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
    "Choose Model",
    options=["LightGBM", "Bagging_Decision_tree", "Vanilla_Net"],
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

    if res == "LightGBM":
        model = joblib.load("ML_models\\PC_LGBMClassifier_BayesSearchCV.model")

    if res == "Bagging_Decision_tree":
        model = joblib.load("ML_models\\PC_BaggingClassifier_baseline.model")

    if res == "Vanilla_Net":

        btn = st.button("predict")

        if btn:
            with st.spinner("Making call to the served TF model....."):
                preds = []

                scaler = joblib.load("ML_models\\NN_scaler.scaler")
                inst_scaled = scaler.transform(df.values)

                prog_bar = st.progress(0)

                for i, ins in enumerate(inst_scaled):
                    pred = predict(instances=ins.tolist())
                    pred = pred["predictions"]
                    idx = tf.argmax(pred, axis=1)

                    prog_bar.progress(i + 1)

                    st.write(
                        f"The predicted class is: {(idx.numpy()[0], idx2class[idx.numpy()[0]])}"
                    )

    if model:

        btn = st.button("predict")

        if res == "LightGBM":
            X = df.values

        elif res == "Bagging_Decision_tree":
            X = df.values

        if btn:
            with st.spinner("Classifying...."):
                preds = model.predict(X)

                for idx, pred in enumerate(preds):
                    st.write(
                        f"The predicted class for instance is: {(pred, idx2class[pred])}"
                    )
