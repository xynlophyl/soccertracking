import os
from utils.vod_utils import read_video, get_video_filename
from tracker.tracker import Tracker
import pickle

def detection_tracking():
    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    CURR_TASK = "detection_tracking"

    input_video = f"{GCP_PROJECT_PATH}/input_videos/08fd33_4.mp4"
    detect_model = f"{GCP_PROJECT_PATH}/models/detect/best.pt"
    vod_frames = read_video(input_video)
    filename = get_video_filename(input_video)
    
    tracker = Tracker(
        model_path=detect_model
    )
    
    # track object across frames
    track_stubs = f"{GCP_PROJECT_PATH}/stubs/track_stubs_{filename}_{CURR_TASK}.pkl"
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        stub_path=track_stubs
    )
        
    print("done detection and tracking.")
    
if __name__ == '__main__':
    detection_tracking()