
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'test_pipeline',
    default_args=default_args,
    description='Test DAG for pipeline validation',
    schedule_interval='@hourly',
    catchup=False,
)

def test_task():
    """Simple test task"""
    print("Pipeline test successful!")
    return "Success"

start = DummyOperator(task_id='start', dag=dag)
test = PythonOperator(
    task_id='test_task',
    python_callable=test_task,
    dag=dag,
)
end = DummyOperator(task_id='end', dag=dag)

start >> test >> end
