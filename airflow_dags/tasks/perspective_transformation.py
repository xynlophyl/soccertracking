import os
import pickle
import numpy as np
from utils.vod_utils import read_video, get_video_filename
from pitchlocalization.view_transformer import ViewTransformer

def perspective_transformation():
    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    PREV_TASK = "keypoint_detection"
    CURR_TASK = "perspective_transformation"

    input_video = f"{GCP_PROJECT_PATH}/input_videos/08fd33_4.mp4"
    vod_frames = read_video(input_video)
    filename = get_video_filename(input_video)
    
    keypoint_stubs = f"{GCP_PROJECT_PATH}/stubs/keypoint_stubs_{filename}.pkl"
    track_stubs = f"{GCP_PROJECT_PATH}/stubs/track_stubs_{filename}.pkl"

    try:
        with open(keypoint_stubs, 'rb') as f:
            all_keypoints = pickle.load(f)
        with open(track_stubs, 'rb') as f:
            tracks = pickle.load(f)
        if all_keypoints and tracks:
            # get pitch keypoints from mplsoccer pitch layout
            mpl_keypoints = np.genfromtxt(f"{GCP_PROJECT_PATH}/assets/mplpitch_keypoints.csv", delimiter=',')[:, 1:]

            # init view transformer
            vt = ViewTransformer(mpl_keypoints, conf = 0.5)
            
            # transform tracking coordinates to 2D plane
            print(tracks)
            tracks = vt.transform_all_points(vod_frames, tracks, all_keypoints)
            
            # saving the intermediate pkl
            with open(f"{GCP_PROJECT_PATH}/stubs/track_stubs_{filename}_{CURR_TASK}.pkl", 'wb') as f:
                pickle.dump(tracks, f)
            
        print("done perspective transformation.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    perspective_transformation()