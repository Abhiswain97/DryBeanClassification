# PGD_thesis

## Serving the tensorflow model

You can directly serve the model and make api calls to get the predictions. Pre-requisite is just have docker installed

- Pull the container: `docker pull tensorflow/serving`

- Run the container: `docker run -it -v C:\Users\abhi0\Projects\PGD_thesis:/DryBean -p 8605:8605 --entrypoint /bin/bash tensorflow/serving`

- Inside the container, start the model server: `tensorflow_model_server --rest_api_port=8605 --model_name=dry_bean_model --model_base_path=/DryBean/models/1/`

## Using the app 

- First, serve the model using the above instructions

- Next, in another terminal first install the requirements using: `pip install -r requirements.txt`

- Now just do: `streamlit run deploy\deploy.py`
