import numpy as np
from pitchlocalization import KeypointDetector, PitchFrame, ViewTransformer
from team_assigner import TeamAssigner
from tracker import Tracker, PlayerBallAssigner
from utils import read_video, save_video, calculate_centroid, measure_distance, get_center_of_bbox
import random

def main():
    
    """
    INITIALIZATION: load video
    """
    print('loading video')
    # vod_frames = read_video("sample_vod.mp4")
    vod_frames = read_video("input_videos/121364_0.mp4")


    """
    TRACKING: initial detection and tracking
    """
    print('adding detections')
    
    # init tracker
    tracker = Tracker(
        # model_path= "./best.pt"
        model_path="./models/detect/best.pt"
    )

    # track object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub = True,
        stub_path = "./stubs/track_stubs_121364_0.pkl"
    )

    """
    TRACKING: BALL INTERPOLATION 
    """
    
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # assign ball to player (if possible)
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # add has_ball property to tracking data if assignment is found
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
    
    """
    TRACKING: team assignment 
    """
    print("assigning team")


    # make sure the team color proportions from k mean makes sense
    team_assigner = TeamAssigner()
    for _ in range(10):
        random_frame_index = random.randrange(len(vod_frames))
        team_color_counter = [0, 0]
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(vod_frames[random_frame_index], tracks['players'][random_frame_index])
        for player_id, track in  tracks['players'][random_frame_index].items():
            if track["cls_name"] == "player":
                team = team_assigner.get_player_team(vod_frames[random_frame_index],   
                                                    track,
                                                    player_id)
                team_color_counter[team-1] += 1

        total_players = sum(team_color_counter)
        smaller_team = min(team_color_counter)
        if smaller_team > total_players * 0.4: # if the smaller team size is at least 40% of the total, the proportions should make sense
            break
        else:
            print("k-mean too biased. re-running")

    # team assignment for goalkeeper 
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if track["cls_name"] == "player":
                team = team_assigner.get_player_team(vod_frames[frame_num],   
                                                    track,
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # estimated the goalkeeper's team based on the teams' centroids
        team_1_player_bboxes = list(map(lambda t: t[1]["bbox"], filter(lambda t: t[1]["cls_name"] == "player" and t[1]["team"] == 1, player_track.items())))
        team_2_player_bboxes = list(map(lambda t: t[1]["bbox"], filter(lambda t: t[1]["cls_name"] == "player" and t[1]["team"] == 2, player_track.items())))

        team_1_centroid = calculate_centroid(team_1_player_bboxes)
        team_2_centroid = calculate_centroid(team_2_player_bboxes)
        for player_id, track in player_track.items():
            if track["cls_name"] == "goalkeeper":
                goalkeeper_bbox = track["bbox"]
                goalkeeper_center = get_center_of_bbox(goalkeeper_bbox)
                dist_1 = measure_distance(goalkeeper_center, team_1_centroid)
                dist_2 = measure_distance(goalkeeper_center, team_2_centroid)
                if dist_1 < dist_2:
                    tracks['players'][frame_num][player_id]['team'] = 1
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[1]
                else:
                    tracks['players'][frame_num][player_id]['team'] = 2
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[2]
    
    """
    TRACKING
    """
    # add annotations to match vod 
    print('adding custom annotations')
    output_frames = tracker.draw_annotations(vod_frames, tracks) 

    # save annotated match vod
    print('saving output')
    save_video(output_frames, "./outputs/output_annotated_vod.avi")

    """
    KEYPOINT DETECTION: initial detection
    """

    kp = KeypointDetector("./models/pose/best.pt")

    # get keypoints
    all_keypoints = kp.get_keypoints(
        vod_frames,
        read_from_stub=True,
        stub_path='pitch_keypoints_stub.pkl'
    )

    """
    PERSPECTIVE TRANSFORM: transform detections (keypoints, tracks)
    """

    # get pitch keypoints from mplsoccer pitch layout
    mpl_keypoints = np.genfromtxt('./assets/mplpitch_keypoints.csv', delimiter=',')[:, 1:]

    # init Pitch
    pitch = PitchFrame()

    for frame_num, frame in enumerate(vod_frames):

        # get pitch keypoint detections for current frame
        frame_keypoints = all_keypoints[frame_num]

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
    minimap_output_frames = pitch.draw_annotations(vod_frames, tracks)

    """
    PERSPECTIVE TRANSFORM: output 2D minimap video
    """
    # save minimap transformation
    save_video(minimap_output_frames, "output_minimamp.avi")

if __name__ == '__main__':

    main()