import os
from utils.vod_utils import read_video, save_video, get_video_filename
from tracker.tracker import Tracker

def detection_tracking():
    gcp_project_path = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    input_video = f"{gcp_project_path}/input_videos/08fd33_4.mp4"
    detect_model = f"{gcp_project_path}/models/detect/best.pt"
    pose_model = f"{gcp_project_path}/models/pose/best.pt"
    vod_frames = read_video(input_video)
    filename = get_video_filename(input_video)
    
    tracker = Tracker(
        model_path=detect_model
    )
    
    # track object across frames
    track_stubs = f"{gcp_project_path}/stubs/track_stubs_{filename}.pkl"
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        stub_path=track_stubs
    )
    
    print("tracks:", tracks)
    
if __name__ == '__main__':
    detection_tracking()