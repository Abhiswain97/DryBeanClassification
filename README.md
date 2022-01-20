# PGD_thesis

## Serving the tensorflow model

- Run the container: `docker run -it -v C:\Users\abhi0\Projects\PGD_thesis:/DryBean -p 8605:8605 --entrypoint /bin/bash tensorflow/serving`

- Inside the container, start the model server: `tensorflow_model_server --rest_api_port=8605 --model_name=dry_bean_model --model_base_path=/DryBean/models/1/`
```
