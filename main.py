
# from tracker import Tracker, PlayerBallAssigner
from tracker import Tracker
from utils import read_video, save_video

def main():
    
    # read video
    print('loading video')
    vod_frames = read_video("sample_vod.mp4")

    print('adding detections')
    # init tracker
    tracker = Tracker(
        model_path= "./best.pt"
    )

    # track detections for each object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        stub_path = "./stubs/track_stubs.pkl"
    ) 

    print('adding custom annotations')
    # add custom ellipse annotations using bbox data
    output_frames = tracker.draw_annotations(vod_frames, tracks) 

    print('saving output')
    # save processed video
    save_video(output_frames, "./outputs/annotated_output.mp4")


if __name__ == '__main__':

    main()