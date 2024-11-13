from tracker import Tracker, PlayerBallAssigner
from utils import read_video, save_video, calculate_centroid, measure_distance, get_center_of_bbox
from team_assigner import TeamAssigner
import random

def main():
    
    # read video
    print('loading video')
    # vod_frames = read_video("sample_vod.mp4")
    vod_frames = read_video("input_videos/121364_0.mp4")


    # TRACKING
    print('adding detections')
    tracker = Tracker(
        # model_path= "./best.pt"
        model_path="./models/best.pt"
    )

    # track detections for each object across frames
    tracks = tracker.get_object_tracks(
        vod_frames,
        read_from_stub = True,
        stub_path = "./stubs/track_stubs_121364_0.pkl"
    )

    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # TEAM ASSIGNMENT
    print("assigning team")
    team_assigner = TeamAssigner()

    # make sure the team color proportions from k mean makes sense
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

    # end of TEAM ASSIGNMENT

    # BALL ASSIGNMENT 
    player_assigner = PlayerBallAssigner()
    # assign ball to player (if possible)
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # add has_ball property to tracking data if assignment is found
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    print('adding custom annotations')
    # add custom ellipse annotations using bbox data
    output_frames = tracker.draw_annotations(vod_frames, tracks) 

    # save processed video
    print('saving output')
    # save_video(output_frames, "./outputs/annotated_output.mp4")
    save_video(output_frames, "./outputs/annotated_output.avi")


if __name__ == '__main__':

    main()