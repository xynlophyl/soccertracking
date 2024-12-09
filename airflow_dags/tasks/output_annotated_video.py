import os
import pickle
from utils.vod_utils import read_video, save_video, get_video_filename
from tracker.tracker import Tracker

def output_annotated_video():
    try:
        GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
        PREV_TASK = "merge_tracks"
        
        input_video_path = f"{GCP_PROJECT_PATH}/input_videos/08fd33_4.mp4"
        detect_model = f"{GCP_PROJECT_PATH}/models/detect/best.pt"
        vod_frames = read_video(input_video_path)
        filename = get_video_filename(input_video_path)
        track_stubs = f"{GCP_PROJECT_PATH}/stubs/track_stubs_{filename}_{PREV_TASK}.pkl"
        
        tracker = Tracker(
            model_path=detect_model
        )

        with open(track_stubs, 'rb') as f:
            tracks = pickle.load(f)
            
            # add annotations to match vod
            print('adding custom annotations')
            output_frames = tracker.draw_annotations(vod_frames, tracks, use_jersey_numbers = False)
            
            # save annotated match vod
            print('saving annotations on vod')
            
            output_video = f"{GCP_PROJECT_PATH}/outputs/output_annotated_{filename}.mp4"
            save_video(output_frames, output_video)
            print("output path:", output_video)

        print("done output annotated video.")
    except Exception as e:
        print(f"Error outputing file: {e}")

if __name__ == '__main__':
    output_annotated_video()