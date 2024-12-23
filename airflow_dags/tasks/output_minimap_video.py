import os
import sys
import pickle
from utils.vod_utils import read_video, save_video
from pitchlocalization.pitch_frame import PitchFrame

def output_minimap_video():
    input_video_path = sys.argv[1]
    filename = sys.argv[2]
    try:
        GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
        PREV_TASK = "merge_tracks"

        vod_frames = read_video(input_video_path)
        track_stubs = f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{PREV_TASK}.pkl"

        with open(track_stubs, 'rb') as f:
            tracks = pickle.load(f)
            
            pitch = PitchFrame()
            minimap_output_frames = pitch.draw_annotations(vod_frames, tracks, use_jersey_numbers = False)

            # save minimap transformation
            print("saving minimap")
            output_minimap = f"{GCP_PROJECT_PATH}/outputs/{filename}/output_minimap_{filename}.avi"
            save_video(minimap_output_frames, output_minimap)
            print("output path:", output_minimap)

        print("done output minimap video.")
    except Exception as e:
        print(f"Error outputing file: {e}")

if __name__ == '__main__':
    output_minimap_video()