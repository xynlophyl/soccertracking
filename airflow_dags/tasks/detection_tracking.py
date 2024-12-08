from utils import read_video, save_video, get_video_filename

def detection_tracking():
    input_video = "/home/wwkb1233/airflow/dags/soccertracking/input_videos/08fd33_4.mp4"
    detect_model = "models/detect/best.pt"
    pose_model = "models/pose/best.pt"
    vod_frames = read_video(input_video)
    print("video frames:", len(vod_frames))

if __name__ == '__main__':
    detection_tracking()