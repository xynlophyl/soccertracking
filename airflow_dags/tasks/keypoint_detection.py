import os
from utils.vod_utils import read_video, get_video_filename
from pitchlocalization.detect import KeypointDetector

def keypoint_detection():
    GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
    CURR_TASK = "keypoint_detection"

    input_video_path = f"{GCP_PROJECT_PATH}/input_videos/08fd33_4.mp4"
    pose_model = f"{GCP_PROJECT_PATH}/models/pose/best.pt"
    vod_frames = read_video(input_video_path)
    filename = get_video_filename(input_video_path)

    kp = KeypointDetector(pose_model)

    # get keypoints
    keypoint_stubs = f"{GCP_PROJECT_PATH}/stubs/keypoint_stubs_{filename}.pkl"
    all_keypoints = kp.get_keypoints(
        vod_frames,
        read_from_stub=True,
        stub_path=keypoint_stubs
    )

    print("done keypoint detection.")
    
if __name__ == '__main__':
    keypoint_detection()