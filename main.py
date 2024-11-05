
from tracker import Tracker
from utils import read_video, save_video

def main():
    
    # read video
    vod_path = ''
    vod_frames = read_video(vod_path)

    # init tracker
    tracker = Tracker("models/best.pt")
    
    # track detections for each object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        stubpath = 'stubs/track_stubs.pkl'
    ) 

    # save processed video
    # save_video(vod_frames, vod_path)

if __name__ == '__main__':

    main()