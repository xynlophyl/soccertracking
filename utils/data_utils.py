import json
import pandas as pd
from utils import get_center_of_bbox, get_bbox_width, get_bbox_height

def load_category_mapping():

    with open('./assets/sn_categories.json', 'r') as f:
        category_mapping = json.load(f)

    return category_mapping

category_mapping = load_category_mapping()

def process_track(track, track_id, id, image_id):
    # get track features
    cls_name = track.get("cls_name")
    bbox = track.get("bbox")
    jersey = track.get("jersey")
    team = track.get("team")

    # create attribute dict
    attributes = {
        "role": cls_name,
        "jersey": jersey,
        "team": team
    }

    # calculate relevant bbox values
    centers = get_center_of_bbox(bbox)
    x, y, _, _ = bbox
    bbox_image = {
        "x": bbox[0], "y": bbox[1],
        "x_center": centers[0], "y_center": centers[1],
        "w": get_bbox_width(bbox), "h": get_bbox_height(bbox)
    }

    track_row = {
        "id": id,
        "image_id": image_id,
        "track_id": track_id,
        "supercategory": category_mapping[cls_name]["supercategory"],
        "category_id": category_mapping[cls_name]["id"],
        "attributes": attributes,
        "bbox_image": bbox_image,
        "bbox_pitch": None,
        "bbox_pitch_raw": None,
        "lines": None,
    }

    return track_row

def reformat_tracks(frames, tracks, save_csv, csv_path):

    num_frames = len(frames)

    player_tracks = tracks["players"]
    referee_tracks = tracks["referees"]
    ball_tracks = tracks["ball"]

    id = 0
    all_rows = []
    for frame_num in range(num_frames):

        rows = []
        # get player track data
        for player_id, track in player_tracks[frame_num].items():
            player_track = process_track(track, player_id, id, frame_num)
            rows.append(player_track)
            id += 1

        for referee_id, track in referee_tracks[frame_num].items():
            track["cls_name"] = "referee"
            referee_track = process_track(track, referee_id, id, frame_num)
            rows.append(referee_track)
            id += 1

        # sort ids in ascending order
        rows = sorted(rows, key = lambda row: row["track_id"])

        for track in ball_tracks[frame_num].values():
            track["cls_name"] = "ball"
            ball_track = process_track(track, None, id, frame_num)
            rows.append(ball_track)
            id += 1

        all_rows += rows

    # create dataframe
    tracks_df = pd.DataFrame(all_rows)

    # assign track id for ball
    ball_id = tracks_df.track_id.max() + 1

    # fill ball's track_id
    tracks_df.loc[tracks_df['category_id'] == 4, "track_id"] = ball_id

    if csv_path is not None:

        tracks_df.to_csv(csv_path)

    return tracks_df