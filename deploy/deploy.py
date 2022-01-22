import streamlit as st
import tensorflow as tf
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict(instances):

    payload = {
        "instances": [
            instances
        ]
    }

    res = requests.post(
        url='http://localhost:8605/v1/models/dry_bean_model:predict', 
        json=payload
    )

    preds = res.json()
    return preds

column_names = [
    'Area',
    'Perimeter',
    'MajorAxisLength',
    'MinorAxisLength',
    'AspectRation',
    'Eccentricity',
    'ConvexArea',
    'EquivDiameter',
    'Extent',
    'Solidity',
    'roundness',
    'Compactness',
    'ShapeFactor1',
    'ShapeFactor2',
    'ShapeFactor3',
    'ShapeFactor4',
    'Class'
]

st.sidebar.markdown("<h1><center>Dry bean classifier</center></h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"<p>The values entered follow the columns in the format: <b><i>{column_names}</i><b></p>", unsafe_allow_html=True)

values = st.text_input(
    f"Enter feature values as comma separated values"
)

btn = st.button("predict")

if btn:
    idx2class = {
        0: 'BARBUNYA',
        1: 'BOMBAY',
        2: 'CALI',
        3: 'DERMASON',
        4: 'HOROZ',
        5: 'SEKER',
        6: 'SIRA'
    }

    values = [float(v) for v in values.split(',')]

    values = StandardScaler().fit_transform(np.array(values).reshape(-1, 1))

    with st.spinner("Classifying...."):
        preds = predict(values.reshape(-1).tolist())
        
        idx = tf.argmax(preds['predictions'], axis=1)

        st.success(f"The predicted class is: {idx2class[idx.numpy()[0]]}")


