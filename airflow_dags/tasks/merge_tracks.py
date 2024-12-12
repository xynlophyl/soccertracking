import os
import sys
import pickle

def merge_dicts(dict1, dict2):
    for track_type in dict1: # players/referees/ball
        for frame_num in range(len(dict1[track_type])):
            dict1_detections = dict1[track_type][frame_num]
            dict2_detections = dict2[track_type][frame_num]
            for track_id in dict1_detections:
                if track_id not in dict2_detections:
                    raise Exception(f"Error: missing track_id {track_id} in dict2")
                dict1_detection = dict1_detections[track_id]
                dict2_detection = dict2_detections[track_id]
                for key in dict2_detection:
                    if key not in dict1_detection:
                        dict1_detection[key] = dict2_detection[key]
    return dict1

def merge_tracks():
    input_video_path = sys.argv[1]
    filename = sys.argv[2]
    try:
        GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
        
        PREV_TASK_1 = "team_assignment"
        track_stubs1 = f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{PREV_TASK_1}.pkl"
        PREV_TASK_2 = "perspective_transformation"
        track_stubs2 = f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{PREV_TASK_2}.pkl"
        
        CURR_TASK = "merge_tracks"
        final_track_stubs = f"{GCP_PROJECT_PATH}/stubs/{filename}/track_stubs_{filename}_{CURR_TASK}.pkl"

        with open(track_stubs1, 'rb') as f1:
            data1 = pickle.load(f1)

        with open(track_stubs2, 'rb') as f2:
            data2 = pickle.load(f2)

        # Merge the data
        merged_data = merge_dicts(data1, data2)

        # Save the merged data
        with open(final_track_stubs, 'wb') as out:
            pickle.dump(merged_data, out)

        print(f"Merged pickle file saved as {final_track_stubs}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    merge_tracks()