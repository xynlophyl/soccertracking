from utils.vod_utils import get_video_filename
from tracker.tracker import Tracker
from tracker.player_ball_assigner import PlayerBallAssigner
import os
import pickle

def ball_interpolation():
    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    PREV_TASK = "detection_tracking"
    CURR_TASK = "ball_interpolation"
    
    input_video = f"{GCP_PROJECT_PATH}/input_videos/08fd33_4.mp4"
    detect_model = f"{GCP_PROJECT_PATH}/models/detect/best.pt"
    filename = get_video_filename(input_video)
    track_stubs = f"{GCP_PROJECT_PATH}/stubs/track_stubs_{filename}_{PREV_TASK}.pkl"
    
    tracker = Tracker(
        model_path=detect_model
    )

    try:
        with open(track_stubs, 'rb') as f:
            tracks = pickle.load(f)

            tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

            # assign ball to player (if possible)
            player_assigner = PlayerBallAssigner()
            tracks = player_assigner.assign_ball_to_player(tracks)
            
            # saving the intermediate pkl
            with open(f"{track_stubs}_${CURR_TASK}", 'wb') as f:
                pickle.dump(tracks, f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == '__main__':
    ball_interpolation()