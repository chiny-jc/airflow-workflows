# Apache Airflow Workflows
This repository contains a collection of personal Airflow workflows.

### 1. ML Pipeline
#### Tasks:
- **download_dataset**: This task includes a bash operator to download a zip file from a website
- **unzip_dataset**: This task includes a bash operator to unzip the previously downloaded file
- **prepare_dataset**: This task includes a Python operator to preprocess the dataset by performing different operations such as dropping some columns, converting the labels to numbers, creating dummy variables, and balancing the dataset. 
- **get_svc_score**: This task includes a Python operator to train a Support Vector Classifier and return the macro F1-score
- **get_knn_score**: This task includes a Python operator to train a K-Nearest-Neighbors Classifier and return the macro F1-score
- **get_rfc_score**: This task includes a Python operator to train a Random Forests Classifier and return the macro F1-score
- **train_best_model**: This task includes a Python operator to obtain the three macro F1-scores from the three previous tasks and select the model with the highest score. Then, it trains the best model with the whole dataset and saves it to a pickle file. 

#### Dependencies: 
- download_dataset -> unzip_dataset
- unzip_dataset -> prepare_dataset
- prepare_dataset -> get_svc_score, get_knn_score, get_rfc_score
- get_svc_score, get_knn_score, get_rfc_score -> train_best_model
