from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add src to path so Airflow can import train_pipeline
sys.path.insert(0, '/opt/airflow/src')

from train_pipeline import prepare_data, preprocess_data, train_and_save_model
from evaluate import evaluate_model

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mental_health_training',
    default_args=default_args,
    description='Prepare, preprocess, train, and evaluate the student mental health model',
    schedule='@weekly',
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'mental_health'],
) as dag:

    prepare_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_and_save_model,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    prepare_task >> preprocess_task >> train_task >> evaluate_task