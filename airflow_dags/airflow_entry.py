from datetime import datetime, timedelta
from textwrap import dedent
import time
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# pip install matplotlib mplsoccer numpy pandas pillow python-dotenv requests roboflow scikit-learn scipy torch torchvision tqdm ultralytics supervision
# pip uninstall opencv-python
# pip install opencv-python-headless

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

    ### tracking stuff

    detection_tracking = BashOperator(
        task_id="detection_tracking",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/detection_tracking.py",
        retries=0,
    )

    ball_interpolation = BashOperator(
        task_id="ball_interpolation",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/ball_interpolation.py",
        retries=0,
    )
    
    team_assignment = BashOperator(
        task_id="team_assignment",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/team_assignment.py",
        retries=0,
    )
    
    output_annotated_video = BashOperator(
        task_id="output_annotated_video",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/output_annotated_video.py",
        retries=0,
    )
    
    ### minimap stuff

    keypoint_detection = BashOperator(
        task_id="keypoint_detection",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/keypoint_detection.py",
        retries=0,
    )

    perspective_transformation = BashOperator(
        task_id="perspective_transformation",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/perspective_transformation.py",
        retries=0,
    )
    
    output_minimap_video = BashOperator(
        task_id="output_minimap_video",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/output_minimap_video.py",
        retries=0,
    )
    
    ### utils
    
    merge_tracks = BashOperator(
        task_id="merge_tracks",
        bash_command="python3 /home/wwkb1233/airflow/dags/soccertracking/airflow_dags/tasks/merge_tracks.py",
        retries=0,
    )

    detection_tracking >> ball_interpolation >> team_assignment
    [keypoint_detection, ball_interpolation] >> perspective_transformation
    [team_assignment, perspective_transformation] >> merge_tracks
    merge_tracks >> [output_annotated_video, output_minimap_video]
    # [output_annotated_video, output_minimap_video] >> output_combined_video
