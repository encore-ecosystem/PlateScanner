from airflow import DAG
from datetime import datetime, timedelta


default_args = {
	'owner': 'airflow',
	'depends_on_past': False,
	'start_date': datetime(2015, 12, 1),
	'email': ['airflow@example.com'],
	'email_on_failure': False,
	'email_on_retry': False,
	'retries': 1,
	'retry_delay': timedelta(minutes=5),
	'schedule_interval': '@hourly',
}

dag = DAG('tutorial', catchup=False, default_args=default_args)