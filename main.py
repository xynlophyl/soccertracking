from tracker import Tracker, PlayerBallAssigner
from utils import read_video, save_video
from team_assigner import TeamAssigner

def main():
    
    # read video
    print('loading video')
    # vod_frames = read_video("sample_vod.mp4")
    vod_frames = read_video("input_videos//08fd33_4.mp4")

    print('adding detections')
    # init tracker
    tracker = Tracker(
        # model_path= "./best.pt"
        model_path="./models/best.pt"
    )

    # track detections for each object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub = True,
        stub_path = "./stubs/track_stubs.pkl"
    )

    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # TEAM ASSIGNMENT
    print("assigning team")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(vod_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(vod_frames[frame_num],   
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assign ball to player (if possible)
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # add has_ball property to tracking data if assignment is found
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    print('adding custom annotations')
    # add custom ellipse annotations using bbox data
    output_frames = tracker.draw_annotations(vod_frames, tracks) 

    print('saving output')
    # save processed video
    # save_video(output_frames, "./outputs/annotated_output.mp4")
    save_video(output_frames, "./outputs/annotated_output.avi")


if __name__ == '__main__':

    main()