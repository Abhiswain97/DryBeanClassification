import streamlit as st
import tensorflow as tf
import requests
import joblib
import pandas as pd


def predict(instances):

    payload = {"instances": [instances]}

    res = requests.post(
        url="https://drybeanapp.herokuapp.com/v1/models/saved_model:predict",
        json=payload,
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
    "<h1><center>Dry Bean Classifier</center></h1>", unsafe_allow_html=True
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


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


if file_uploader is not None:
    df = pd.read_csv(file_uploader)

    st.dataframe(data=df.head())
    model = None
    predictions = []

    if res == "LightGBM":
        model = joblib.load("./ML_models/PC_LGBMClassifier_BayesSearchCV.model")

    if res == "Bagging_Decision_tree":
        model = joblib.load("./ML_models/PC_BaggingClassifier_baseline.model")

    if res == "Vanilla_Net":

        btn = st.button("predict")

        if btn:

            with st.spinner("Making call to the served TF model....."):
                scaler = joblib.load("./ML_models/NN_scaler.scaler")
                inst_scaled = scaler.transform(df.values)

                for i, ins in enumerate(inst_scaled):
                    pred = predict(instances=ins.tolist())

                    pred = pred["predictions"]
                    idx = tf.argmax(pred, axis=1)

                    predictions.append(idx2class[idx.numpy()[0]])

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
                    predictions.append(idx2class[pred])

    if len(predictions) != 0:

        pred_df = pd.DataFrame({"labels": predictions}).reset_index(drop=True)

        st.success("Prediction complete for all instances")
        st.dataframe(pred_df)

        csv = convert_df(pred_df)

        st.download_button(
            "Press to Download", csv, "preds.csv", "text/csv", key="download-csv"
        )
