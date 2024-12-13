import os
from datetime import datetime, timedelta
from textwrap import dedent
from utils.vod_utils import get_video_filename
import time
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# pip install matplotlib mplsoccer numpy pandas pillow python-dotenv requests roboflow scikit-learn scipy torch torchvision tqdm ultralytics supervision
# pip uninstall opencv-python
# pip install opencv-python-headless

default_args = {
    "owner": "yy3223",
    "depends_on_past": False,
    "email": ["yy3223@columbia.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    "eecs6893-soccer-project-gcp",
    default_args=default_args,
    description="eecs6893 soccer project",
    start_date=datetime(2021, 1, 1),
    params={"filename": "input"},
    catchup=False,
    tags=["eecs6893"],
) as dag:
    # gsutil cp gs://eecs6893-yy3223/inputs/08fd33_4.mp4 /home/wwkb1233/airflow/dags/soccertracking/input_videos

    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    filename = "{{ params.filename }}"
    input_video_path = f"{GCP_PROJECT_PATH}/input_videos/{filename}.mp4"

    ### tracking stuff

    detection_tracking = BashOperator(
        task_id="detection_tracking",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/detection_tracking.py {input_video_path} {filename}",
        retries=0,
    )

    ball_interpolation = BashOperator(
        task_id="ball_interpolation",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/ball_interpolation.py {input_video_path} {filename}",
        retries=0,
    )
    
    team_assignment = BashOperator(
        task_id="team_assignment",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/team_assignment.py {input_video_path} {filename}",
        retries=0,
    )
    
    output_annotated_video = BashOperator(
        task_id="output_annotated_video",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/output_annotated_video.py {input_video_path} {filename}",
        retries=0,
    )
    
    ### minimap stuff

    keypoint_detection = BashOperator(
        task_id="keypoint_detection",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/keypoint_detection.py {input_video_path} {filename}",
        retries=0,
    )

    perspective_transformation = BashOperator(
        task_id="perspective_transformation",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/perspective_transformation.py {input_video_path} {filename}",
        retries=0,
    )
    
    output_minimap_video = BashOperator(
        task_id="output_minimap_video",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/output_minimap_video.py {input_video_path} {filename}",
        retries=0,
    )
    
    ### utils
    
    transfer_input_file = BashOperator(
        task_id="transfer_input_file",
        bash_command=f"gsutil cp gs://eecs6893-yy3223/inputs/{filename}.mp4 {GCP_PROJECT_PATH}/input_videos && mkdir -p {GCP_PROJECT_PATH}/stubs/{filename} && mkdir -p {GCP_PROJECT_PATH}/outputs/{filename}",
        retries=0,
    )
    
    merge_tracks = BashOperator(
        task_id="merge_tracks",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/merge_tracks.py {input_video_path} {filename}",
        retries=0,
    )
    
    output_combined_video = BashOperator(
        task_id="output_combined_video",
        bash_command=f"python3 {GCP_PROJECT_PATH}/airflow_dags/tasks/output_combined_video.py {input_video_path} {filename}",
        retries=0,
    )

    transfer_output_files = BashOperator(
        task_id="transfer_output_files",
        bash_command=f"gsutil cp -r {GCP_PROJECT_PATH}/outputs/{filename} gs://eecs6893-yy3223/outputs && gsutil cp -r {GCP_PROJECT_PATH}/stubs/{filename} gs://eecs6893-yy3223/outputs",
        retries=0,
    )

    transfer_input_file >> [detection_tracking, keypoint_detection]
    detection_tracking >> ball_interpolation >> team_assignment
    [keypoint_detection, ball_interpolation] >> perspective_transformation
    [team_assignment, perspective_transformation] >> merge_tracks
    merge_tracks >> [output_annotated_video, output_minimap_video]
    [output_annotated_video, output_minimap_video] >> output_combined_video
    output_combined_video >> transfer_output_files
