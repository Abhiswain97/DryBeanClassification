# PGD_thesis

## Table of Contents:

View the noteboks phase-wise following the links:

- [Data Acquisition](https://nbviewer.org/github/Abhiswain97/PGD_thesis/blob/master/Phase-1-Documentation-notebook.ipynb) 
- [EDA - Exploratory Data Analysis](https://nbviewer.org/github/Abhiswain97/PGD_thesis/blob/master/Phase-2-EDA-notebook.ipynb)
- [Feature Selection & Modelling](https://nbviewer.org/github/Abhiswain97/PGD_thesis/blob/master/Phase-3-Feature-Selection-and-Modelling%20.ipynb)
- [Advanced Modelling](https://nbviewer.org/github/Abhiswain97/PGD_thesis/blob/master/Phase-4-TF-NN.ipynb)


## Serving the tensorflow model

You can directly serve the model and make api calls to get the predictions. Pre-requisite is just have docker installed

- Pull the container: `docker pull tensorflow/serving`

- Run the container: `docker run -it -v C:\Users\abhi0\Desktop\PGD_thesis:/DryBean -p 8605:8605 --entrypoint /bin/bash tensorflow/serving`

- Inside the container, start the model server: `tensorflow_model_server --rest_api_port=8605 --model_name=dry_bean_model --model_base_path=/DryBean/models/best_model`

## Using the app 

- First, serve the model using the above instructions

- Next, in another terminal first install the requirements using: `pip install -r requirements.txt`

- Now just do: `streamlit run deploy\deploy.py`
