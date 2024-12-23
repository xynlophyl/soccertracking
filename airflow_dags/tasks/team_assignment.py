from utils.vod_utils import read_video
import os
import pickle
import sys
from team_assigner.team_assigner import TeamAssigner

def team_assignment():
    input_video_path = sys.argv[1]
    filename = sys.argv[2]

    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    PREV_TASK = "ball_interpolation"
    CURR_TASK = "team_assignment"
    
    vod_frames = read_video(input_video_path)
    track_stubs = f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{PREV_TASK}.pkl"

    try:
        with open(track_stubs, 'rb') as f:
            tracks = pickle.load(f)

            team_assigner = TeamAssigner()

            # make sure the team color proportions from k mean makes sense
            tracks = team_assigner.assign_team_by_sampling(vod_frames, tracks)

            # team assignment for goalkeeper 
            tracks = team_assigner.get_goalkeeper_team(vod_frames, tracks)
            
            # saving the intermediate pkl
            with open(f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{CURR_TASK}.pkl", 'wb') as f:
                pickle.dump(tracks, f)

        print("done team assignment.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == '__main__':
    team_assignment()