import streamlit as st
import tensorflow as tf
import requests
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dry Bean app",
    page_icon=":seedling:",
    layout="wide",
)


st.sidebar.image("images/Beans.png")
# Hide hamburger icon and made with streamlit

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

idx2class = {
    0: "BARBUNYA",
    1: "BOMBAY",
    2: "CALI",
    3: "DERMASON",
    4: "HOROZ",
    5: "SEKER",
    6: "SIRA",
}


@st.cache
def pred_NN(X):
    predictions = []
    confs = []

    with st.spinner("Making call to the served TF model....."):
        scaler = joblib.load("./ML_models/NN_scaler.scaler")
        inst_scaled = scaler.transform(X)

        for i, ins in enumerate(inst_scaled):
            payload = {"instances": [ins.tolist()]}

            res = requests.post(
                url="https://drybeanapp.herokuapp.com/v1/models/saved_model:predict",
                json=payload,
            )

            pred = res.json()

            pred = pred["predictions"]

            confidence = tf.nn.softmax(pred[0])

            idx = tf.argmax(pred, axis=1)

            predictions.append(idx2class[idx.numpy()[0]])

            confs.append(np.max(confidence) * 100)

            pred_df = pd.DataFrame({"labels": predictions, "confidence": confs})

        return pred_df


@st.cache
def load_model(model_name):

    model = None

    if model_name == "LightGBM":

        model = joblib.load("./ML_models/PC_LGBMClassifier_BayesSearchCV.model")

    elif model_name == "Ensemble-DT":

        model = joblib.load("./ML_models/PC_BaggingClassifier_baseline.model")

    else:
        raise NotImplementedError("Model not implemented")

    return model


@st.cache
def predict(feats, model):
    predictions = []
    probs = []
    with st.spinner("Classifying...."):
        preds = model.predict(feats)
        prob = model.predict_proba(feats)
        for idx, pred in enumerate(preds):
            predictions.append(idx2class[pred])
            probs.append(np.max(prob) * 100)

    pred_df = pd.DataFrame({"labels": predictions, "confidence": probs})

    return pred_df


# ----------------------------------------- UI ---------------------------------------------

# Sidebars
# Title "Dry bean Classifier"

st.sidebar.title("Dry Bean Classifier")

# Type of predicition
pred_type = st.sidebar.selectbox(
    "Type of predition", options=["Single", "Batch"], index=0
)

# Choose model
model_type = st.sidebar.selectbox(
    "Choose Model", options=["LightGBM", "Ensemble-DT", "Vanilla-Net"], index=0
)

st.sidebar.markdown(
    "<h3>App for Classification of Dry Beans from shape and dimensional features</h3>",
    unsafe_allow_html=True,
)
pred_df = pd.DataFrame()

# Single prediction done using a form
model = None

if pred_type == "Single":

    with st.form("Dry Bean Classification", clear_on_submit=True):
        st.markdown(
            "<h1><center>Enter Feature values</center></h1>",
            unsafe_allow_html=True,
        )

        r1 = st.columns(4)
        r2 = st.columns(4)
        r3 = st.columns(4)
        r4 = st.columns(4)
        r5 = st.columns(5)

        # Row 1
        Area = r1[0].text_input("Area", value="40000")
        Perimeter = r1[1].text_input("Perimeter", value="727.877")
        MajorAxisLength = r1[2].text_input("MajorAxisLength", value="246.6991625")
        MinorAxisLength = r1[3].text_input("MinorAxisLength", value="206.8884621")

        # Row 2
        AspectRatio = r2[0].text_input("AspectRatio", value="1.192425909")
        Eccentricity = r2[1].text_input("Eccentricity", value="0.544706845")
        ConvexArea = r2[2].text_input("ConvexArea", value="40425")
        EquiDiameter = r2[3].text_input("EquiDiameter", value="225.6758334")

        # Row 3
        Extent = r3[0].text_input("Extent", value="0.755857899")
        Solidity = r3[1].text_input("Solidity", value="0.989486704")
        Roundness = r3[2].text_input("Roundness", value="0.94875453")
        Compactness = r3[3].text_input("Compactness", value="0.914781514")

        # Row 4
        ShapeFactor1 = r4[0].text_input("ShapeFactor1", value="0.006167479")
        ShapeFactor2 = r4[1].text_input("ShapeFactor2", value="0.00266414")
        ShapeFactor3 = r4[2].text_input("ShapeFactor3", value="0.836825218")
        ShapeFactor4 = r4[3].text_input("ShapeFactor4", value="0.997852072")

        submit_res = r5[2].form_submit_button("Predict")

        feats = [
            Area,
            Perimeter,
            MajorAxisLength,
            MinorAxisLength,
            AspectRatio,
            Eccentricity,
            ConvexArea,
            EquiDiameter,
            Extent,
            Solidity,
            Roundness,
            Compactness,
            ShapeFactor1,
            ShapeFactor2,
            ShapeFactor3,
            ShapeFactor4,
        ]

        if submit_res:
            count = 0
            for feat in feats:
                if feat == "":
                    count += 1

            if count != 0:
                st.error(
                    "One or more fields are left blank! Filling it with default value!"
                )
            else:
                try:
                    feats = [float(feat) for feat in feats]
                except:
                    st.error(
                        "Only int or float values are allowed! Filling with default values!"
                    )
                    st.stop()

                if model_type == "Vanilla-Net":
                    pred_df = pred_NN(X=[feats])
                    st.markdown(
                        f"<h2><center>The predicted class is: {pred_df.labels[0]} with a confidence of: {pred_df.confidence[0]}</center></h2>",
                        unsafe_allow_html=True,
                    )

                else:
                    try:
                        model = load_model(model_name=model_type)
                    except:
                        print("Reloading model")
                    finally:
                        model = load_model(model_name=model_type)

                    pred_df = predict(feats=[feats], model=model)

                    st.markdown(
                        f"<h2><center>The predicted class is: {pred_df.labels.values[0]} with a confidence of: {pred_df.confidence[0]}</center></h2>",
                        unsafe_allow_html=True,
                    )

    if len(pred_df) != 0:
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download predictions", data=csv, file_name="preds.csv"
        )

else:
    st.markdown(
        "<h3><center>Upload test.csv file</center></h3>", unsafe_allow_html=True
    )

    url = "https://feat-files.s3.us-east-2.amazonaws.com/full_feats_test.csv"
    st.markdown(
        f"<center><i>For testing the batch prediciton module download the csv file from <a href={url}>here</a><center> and upload it</i></center>",
        unsafe_allow_html=True,
    )

    file_uploader = st.file_uploader("")

    if file_uploader:
        df = pd.read_csv(file_uploader)

        st.dataframe(data=df.head())

        r5 = st.columns(5)
        btn = r5[2].button("Predict")

        if btn:
            if model_type == "Vanilla-Net":
                pred_df = pred_NN(X=df.values)

                st.dataframe(pred_df)
            else:
                try:
                    model = load_model(model_name=model_type)
                except:
                    print("Reloading model")
                finally:
                    model = load_model(model_name=model_type)

                pred_df = predict(feats=df.values, model=model)

                st.dataframe(pred_df)
        if len(pred_df) != 0:
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download predictions", data=csv, file_name="preds.csv"
            )
