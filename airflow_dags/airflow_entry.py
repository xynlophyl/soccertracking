from datetime import datetime, timedelta
from textwrap import dedent
import time
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# pip install matplotlib mplsoccer numpy opencv-python pandas pillow python-dotenv requests roboflow scikit-learn scipy torch torchvision tqdm ultralytics

count = 0


def correct_sleeping_function():
    pass


default_args = {
    "owner": "yy3223",
    "depends_on_past": False,
    "email": ["yy3223@columbia.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    "eecs6893-soccer-project-gcp",
    default_args=default_args,
    description="eecs6893 soccer project",
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["eecs6893"],
) as dag:

    # t4 = PythonOperator(
    #     task_id='t4',
    #     python_callable=correct_sleeping_function,
    # )
    
    # gsutil cp gs://eecs6893-yy3223/inputs/08fd33_4.mp4 /home/wwkb1233/airflow/dags/soccertracking/input_videos

    detection_tracking = BashOperator(
        task_id="detection_tracking",
        bash_command="python3 /home/wwkb1233/airflow/dags/tasks/detection_tracking.py",
        retries=1,
    )

    ball_interpolation = BashOperator(
        task_id="ball_interpolation",
        bash_command="python3 /home/wwkb1233/airflow/dags/tasks/ball_interpolation.py",
        retries=1,
    )

    detection_tracking >> ball_interpolation
    # t1 >> [t2, t3, t4, t5]
    # t2 >> t6
    # t3 >> [t7, t12]
    # t5 >> [t8, t9]
    # t7 >> t13
    # t8 >> [t10, t15]
    # t9 >> [t11, t12]
    # [t7, t10, t11, t12] >> t14
    # [t7, t13, t15, t17] >> t18
