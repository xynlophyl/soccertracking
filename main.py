import numpy as np
import torch

from pitchlocalization import KeypointDetector, PitchFrame, ViewTransformer
from team_assigner import TeamAssigner
from tracker import Tracker, PlayerBallAssigner
# from jersey_recognition import JerseyOCR
from utils import read_video, save_video, get_video_filename

def main(
    input_video = "input_videos/08fd33_4.mp4",
    detect_model = "models/detect/best.pt",
    pose_model = "models/pose/best.pt",
):

    """
    INITIALIZATION: load video
    """
    print('loading video')
    vod_frames = read_video(input_video)

    filename = get_video_filename(input_video)

    """
    TRACKING: initial detection and tracking
    """
    print('adding detections')
    
    # init tracker
    tracker = Tracker(
        model_path=detect_model
    )

    # track object across frames
    track_stubs = f"stubs/track_stubs_{filename}.pkl"
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        stub_path=track_stubs
    )

    """
    TRACKING: BALL INTERPOLATION 
    """
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # assign ball to player (if possible)
    player_assigner = PlayerBallAssigner()
    tracks = player_assigner.assign_ball_to_player(tracks)
    
    """
    TRACKING: team assignment 
    """
    print("assigning team")
    team_assigner = TeamAssigner()

    # make sure the team color proportions from k mean makes sense
    tracks = team_assigner.assign_team_by_sampling(vod_frames, tracks)

    # team assignment for goalkeeper 
    tracks = team_assigner.get_goalkeeper_team(vod_frames, tracks)

    """
    TRACKING: jersey number recognition
    """
    # ocr = JerseyOCR(batch_size = 8, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    jersey_stubs = f"./stubs/jersey_track_stubs_{filename}.pkl"
    if False:
        tracks = ocr.get_jersey_tracks(
            vod_frames, 
            tracks, 
            stub_path= jersey_stubs
        )
    else:
        tracks = tracker.get_object_tracks(
            vod_frames,
            read_from_stub=True,
            stub_path=jersey_stubs
        )
    
    """
    TRACKING: annotations
    """
    # add annotations to match vod 
    print('adding custom annotations')
    output_frames = tracker.draw_annotations(vod_frames, tracks, use_jersey_numbers = False) 

    # # save annotated match vod
    print('saving annotations on vod')
    
    output_video = f"./outputs/output_annotated_{filename}.avi"
    save_video(output_frames, output_video)

    """
    KEYPOINT DETECTION: initial detection
    """

    print("key point")
    kp = KeypointDetector(pose_model)

    # get keypoints
    keypoint_stubs = f"./stubs/keypoint_stubs_{filename}.pkl"
    all_keypoints = kp.get_keypoints(
        vod_frames,
        read_from_stub=True,
        stub_path=keypoint_stubs
    )

    """
    PERSPECTIVE TRANSFORM: transform detections (keypoints, tracks)
    """

    print('perspective transform')
    # get pitch keypoints from mplsoccer pitch layout
    mpl_keypoints = np.genfromtxt('./assets/mplpitch_keypoints.csv', delimiter=',')[:, 1:]

    # init view transformer
    vt = ViewTransformer(mpl_keypoints, conf = 0.5)
    
    # transform tracking coordinates to 2D plane
    tracks = vt.transform_all_points(vod_frames, tracks, all_keypoints)

    """
    PERSPECTIVE TRANSFORM: annotate on 2D minimap 
    """
    print("draw minimap")
    pitch = PitchFrame()
    minimap_output_frames = pitch.draw_annotations(vod_frames, tracks, use_jersey_numbers = False)

    """
    PERSPECTIVE TRANSFORM: output 2D minimap video
    """
    # save minimap transformation
    print("saving minimap")
    output_minimap = f"./outputs/output_minimap_{filename}.avi"
    save_video(minimap_output_frames, output_minimap)

if __name__ == '__main__':

    main()