# PGD_thesis

## What is the problem about?

  Dry bean is the most popular pulse produced in the world. The main problem dry bean
  producers and marketers face is in ascertaining good seed quality. Lower quality of
  seeds leads to lower quality of produce. Seed quality is the key to bean cultivation in
  terms of yield and disease. Manual classification and sorting of bean seeds is a difficult
  process. Our objective is to use Machine learning techniques to do the automatic
  classification of seeds.

## Why is this problem important to solve?

  Ascertaining seed quality is important for producers and marketers. Doing this manually
  would require a lot of effort and is a difficult process. This is why we try to use machine
  learning techniques to do the automatic classification of seeds.
  
## Business/Real-World impact of solving this problem?
  - Saves hours of manual sorting and classification of seeds.
  - We can do it in real-time.


## Table of Contents:

View the noteboks phase-wise following the links:

- [Data Acquisition](https://github.com/Abhiswain97/PGD_thesis/blob/master/Phase-1-Documentation-notebook.ipynb) 
- [EDA - Exploratory Data Analysis](https://github.com/Abhiswain97/PGD_thesis/blob/master/Phase-2-EDA.ipynb)
- [Feature Selection & Modelling](https://github.com/Abhiswain97/PGD_thesis/blob/master/Phase-3-Feature-Selection-and-Modelling%20.ipynb)
- [Advanced Modelling](https://github.com/Abhiswain97/PGD_thesis/blob/master/Phase-4-TF-NN.ipynb)

## Results

- The best model we have is a *Light Gradient Boosting Classifer* on data with fixed imabalance. It has an accuracy of 93 % and a F1-score of 0.929

<p align="center">
  <img src="ML_results/CF_Transformed_Tuned_LGBMClassifier.png">
</p>

- The second best model we have is a simple *Light Gradient Boosting Classifer* without any transforms. It has an accuracy of 92.72 % and a f1-score of 0.927

<p align="center">
  <img src="ML_results/CF_LGBMClassifier.png">
</p>


## Serving the tensorflow model

You can directly serve the model and make api calls to get the predictions. Pre-requisite is just have docker installed

- Pull the container: `docker pull tensorflow/serving`

- Run the container: `docker run -it -v C:\Users\abhi0\Desktop\PGD_thesis:/DryBean -p 8605:8605 --entrypoint /bin/bash tensorflow/serving`

- Inside the container, start the model server: `tensorflow_model_server --rest_api_port=8605 --model_name=dry_bean_model --model_base_path=/DryBean/models/best_model`

## Using the app

<p align="center">
  <img src="images/app.png">
</p>


- First, serve the model using the above instructions

- Next, in another terminal first install the requirements using: `pip install -r requirements.txt`

- Now just do: `streamlit run deploy\deploy.py`


