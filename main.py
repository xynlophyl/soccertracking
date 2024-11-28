import numpy as np
from pitchlocalization import KeypointDetector, PitchFrame, ViewTransformer
from team_assigner import TeamAssigner
from tracker import Tracker, PlayerBallAssigner
from utils import read_video, save_video

def main():
    
    """
    INITIALIZATION: load video
    """
    print('loading video')
    vod_frames = read_video("input_videos/sample_vod.mp4")
    # vod_frames = read_video("input_videos/121364_0.mp4")


    """
    TRACKING: initial detection and tracking
    """
    print('adding detections')
    
    # init tracker
    tracker = Tracker(
        model_path="./models/detect/best.pt"
    )

    # track object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub=True,
        # stub_path = "./stubs/track_stubs_121364_0.pkl"
        stub_path='./stubs/track_stubs.pkl'
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
    TRACKING: annotations
    """
    # add annotations to match vod 
    print('adding custom annotations')
    output_frames = tracker.draw_annotations(vod_frames, tracks) 

    # # save annotated match vod
    print('saving annotations on vod')
    save_video(output_frames, "./outputs/output_annotated_vod.avi")

    """
    KEYPOINT DETECTION: initial detection
    """

    print("key point")
    kp = KeypointDetector("./models/pose/best.pt")

    # get keypoints
    all_keypoints = kp.get_keypoints(
        vod_frames,
        read_from_stub=True,
        stub_path='./stubs/pitch_keypoints_stub.pkl'
    )

    """
    PERSPECTIVE TRANSFORM: transform detections (keypoints, tracks)
    """

    # get pitch keypoints from mplsoccer pitch layout
    mpl_keypoints = np.genfromtxt('./assets/mplpitch_keypoints.csv', delimiter=',')[:, 1:]

    # init Pitch
    print("pitch")
    pitch = PitchFrame()

    for frame_num, frame in enumerate(vod_frames):

        # get pitch keypoint detections for current frame
        frame_keypoints = all_keypoints[frame_num][0]

        # get transform
        vt = ViewTransformer(frame_keypoints, mpl_keypoints)

        # get player and ball tracks for current frame
        player_tracks = tracks['players'][frame_num]
        ball_tracks = tracks['ball'][frame_num]

        # get player and ball position coordinates using bounding boxes
        player_positions = [p['bbox'] for p in player_tracks.values()]
        ball_positions = [b['bbox'] for b in ball_tracks.values()]

        player_positions = np.array([((x1+x2)/2, y2) for (x1,y1,x2,y2) in player_positions])
        ball_positions = np.array([((x1+x2)/2, y2) for (x1,y1,x2,y2) in ball_positions])

        # transform object positions to 2D plane
        transformed_player_positions = vt.transform_points(player_positions)
        transformed_ball_positions = vt.transform_points(ball_positions)

        # update tracks
        for idx, p in enumerate(player_tracks):
            tracks['players'][frame_num][p]['xy_2D'] = transformed_player_positions[idx]

        for idx, b in enumerate(ball_tracks):
            tracks['ball'][frame_num][b]['xy_2D'] = transformed_ball_positions[idx]


    """
    PERSPECTIVE TRANSFORM: annotate on 2D minimap 
    """
    print("draw minimap")
    minimap_output_frames = pitch.draw_annotations(vod_frames, tracks)

    """
    PERSPECTIVE TRANSFORM: output 2D minimap video
    """
    # save minimap transformation
    print("saving minimap")
    save_video(minimap_output_frames, "./outputs/output_minimap_121364_0.avi")

if __name__ == '__main__':

    main()