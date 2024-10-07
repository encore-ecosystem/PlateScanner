from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

default_args = {
    'owner': 'airflow',
    'retries': 0,
}


with DAG(
    dag_id       = "Preprocess Data",
    default_args = default_args,
    params       = {
        "epochs"    :       30,
        "batch"     :        8,
        "imgsz"     :     1280,
        "yolo_type" : "YOLOv8",
        "data_id"   : None,
    },
) as dag:

    @task
    def test():
        context = get_current_context()
        print(context)

    test()
