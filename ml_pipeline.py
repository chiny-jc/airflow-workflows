import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def _prepare_dataset():
    dataset = pd.read_csv('~/airflow-docker/downloads/dataset_diabetes/diabetic_data.csv')
    dataset = dataset.drop(['encounter_id', 'patient_nbr'], axis=1)
    dataset = dataset.replace(to_replace='?', value=np.NaN)
    dataset = dataset.dropna(axis=1)
    dataset = _labels_to_numbers(dataset, 'readmitted')
    dataset = pd.get_dummies(dataset, drop_first=True)
    dataset = _balance_dataset(dataset, 'readmitted')
    dataset.to_csv('~/airflow-docker/downloads/clean_dataset.csv', index=False)

def _labels_to_numbers(dataset, column):
    labels = dataset[column].unique()
    num_labels = len(labels)
    label_dict = dict(zip(labels, range(0, num_labels)))
    numerical_labels = dataset[column].map(label_dict)
    return dataset.assign(**{column: numerical_labels})

def _balance_dataset(dataset, target_column):
    value_counts = dataset[target_column].value_counts()
    min_count = min(value_counts)
    return dataset.groupby(target_column).sample(n=min_count, random_state=42)

def _get_model_score(filepath, target_column, model):
    X, y = _split_dataset(filepath, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

def _split_dataset(filepath, target_column):
    dataset = pd.read_csv(filepath)
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    return X, y

def _train_best_model(ti, filepath, target_column):
    scores = ti.xcom_pull(task_ids=[
        'get_svc_score',
        'get_knn_score',
        'get_rfc_score'
    ])

    model, model_name = _select_best_model(scores)
    X, y = _split_dataset(filepath, target_column)
    model.fit(X, y)
    with open(f'/home/joshua/airflow-docker/downloads/{model_name}_model.pk', 'wb') as f:
        pickle.dump(model, f)
    
    return max(scores)

def _select_best_model(scores):
    best_model = np.argmax(np.array(scores))
    if best_model == 0:
        return SVC(), 'svc'
    elif best_model == 1:
        return KNeighborsClassifier(), 'knn'
    else:
        return RandomForestClassifier(), 'rfc'


with DAG(
    dag_id='ml_pipeline',
    schedule_interval='@monthly',
    start_date=datetime(2022, 1, 1),
    catchup=False
) as dag:

    download_dataset = BashOperator(
        task_id='download_dataset',
        bash_command='curl -o ~/airflow-docker/downloads/dataset_diabetes.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip'
    )

    unzip_dataset = BashOperator(
        task_id='unzip_dataset',
        bash_command='unzip ~/airflow-docker/downloads/dataset_diabetes.zip  -d ~/airflow-docker/downloads/'
    )

    prepare_dataset = PythonOperator(
        task_id='prepare_dataset',
        python_callable=_prepare_dataset
    )

    get_svc_score = PythonOperator(
        task_id='get_svc_score',
        python_callable=_get_model_score,
        op_kwargs={
            'filepath': '~/airflow-docker/downloads/clean_dataset.csv', 
            'target_column': 'readmitted', 
            'model': SVC()}
    )

    get_knn_score = PythonOperator(
        task_id='get_knn_score',
        python_callable=_get_model_score,
        op_kwargs={
            'filepath': '~/airflow-docker/downloads/clean_dataset.csv', 
            'target_column': 'readmitted', 
            'model': KNeighborsClassifier()}
    )

    get_rfc_score = PythonOperator(
        task_id='get_rfc_score',
        python_callable=_get_model_score,
        op_kwargs={
            'filepath': '~/airflow-docker/downloads/clean_dataset.csv', 
            'target_column': 'readmitted', 
            'model': RandomForestClassifier()}
    )

    train_best_model = PythonOperator(
        task_id='train_best_model',
        python_callable=_train_best_model,
        op_kwargs={
            'filepath': '~/airflow-docker/downloads/clean_dataset.csv', 
            'target_column': 'readmitted'
        }
    )

    download_dataset >> unzip_dataset >> prepare_dataset >> [get_svc_score, get_knn_score, get_rfc_score] >> train_best_model
    