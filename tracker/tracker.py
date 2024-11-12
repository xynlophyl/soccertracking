import cv2
import numpy as np
import os
import pandas as pd
import pickle
import supervision as sv
from typing import Optional
from ultralytics import YOLO
import sys
sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox

# TODO: add type hints

class Tracker():

    """
    object detection model with tracking
    """

    def __init__(self, model_path: str) -> None:
        
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames: list, batch_size: int = 32) -> list:
        """
        get detection bboxes using trained YOLO model 
        """

        all_detections = []

        # iterate over frames in batches
        for i in range(0, len(frames), batch_size):
            # get object detections for current batch
            detections = self.model.predict(
                frames[i:i+batch_size], 
                conf=0.1
                )
            
            all_detections += detections
        
        return all_detections

    def get_object_tracks(
            self,
            frames: list,
            remap_gk: bool = True,
            read_from_stub: bool = False,
            stub_path: str = None
        ):

        """
        assigns tracking for each detected object in video across across each frame, with conditional remapping of goalkeepers to general player class
        """

        # if stub already exists, return
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)

            return tracks

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # convert to supervision detction format
            detections_sv = sv.Detections.from_ultralytics(detection)

            if remap_gk:
                # convert goalkeeper to player object
                for obj_idx, class_id in enumerate(detections_sv.class_id):
                    pass
                    # if cls_names[class_id] == "goalkeeper":
                    #     detections_sv.class_id[obj_idx] = cls_names_inv["player"]

                # update detection mappings
                detection_with_tracks = self.tracker.update_with_detections(detections_sv)

            else:
                detection_with_tracks = detections_sv

            # add current bbox detections of each object according to their track_id
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # tracking players and referees in each frame
            for frame_detection in detection_with_tracks:

                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["goalkeeper"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "cls_name": "goalkeeper"}

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "cls_name": "player"}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # tracking ball
            for frame_detection in detections_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # save tracking data
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def interpolate_ball(self, ball_positions: list) -> list:

        """
        estimate untracked ball positions using linear interpolation and backfilling
        """

        ball_positions = [
            x.get(1, {}).get('bbox',[]) for x in ball_positions
        ]
        
        # interpolate using pandas
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # convert back to array of dictionaries
        updated_ball_positions = [{1:{"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return updated_ball_positions
      
    def draw_ellipse(
            self, 
            frame: list, 
            bbox: list, 
            color: tuple, 
        ):

        """
        annotates each frame with ellipse to highlight around detected objects
        """

        # get bottom y and center x of bbox
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)

        # get width of bbox
        width = get_bbox_width(bbox)

        # draw ellipse around bottom of detected object
        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        return frame

    def draw_rectangle(
            self,
            frame: list,
            bbox: list,
            color: list,
            track_id: Optional[int] = None
    ):
        
        """
        annotates each frame with rectangle to indicate detected object's tracking id
        """

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        
        rect_width = 40
        rect_height = 20

        # calculate rectangle dimensions
        x1_rect = x_center - rect_width//2
        x2_rect = x_center + rect_width//2
        y1_rect = (y2 - rect_height//2)+15
        y2_rect = (y2 + rect_height//2)+15

        if track_id is not None:
            # draw rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # add tracked object's id as text
            cv2.putText(
                frame, 
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_triangle(
            self,
            frame: list,
            bbox: list,
            color: list
    ):
        
        """
        annotates each frame with triangle above detected object
        """

        # get top y and center x of bbox 
        y1= int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        # calulate coordinates of triangle annotation
        triangle_points = np.array([
            [x_center, y1],
            [x_center-10, y1-20],
            [x_center+10, y1-20]
        ])

        # draw triangle
        cv2.drawContours(
            frame, 
            [triangle_points],
            0, 
            color,
            cv2.FILLED
         )
        
        # draw border around triangle
        cv2.drawContours(
            frame, 
            [triangle_points],
            0, 
            (0,0,0),
            2
        )
        
        return frame
        
    def draw_annotations(self, frames: list, tracks: list):

        """
        draw custom annotations using bbox detections
        TODO: after segmenting between teams of players, add colors for each different team (maybe the color of their kits?)
        """

        annotated_frames = []

        for frame_num, frame in enumerate(frames):

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # add annotation for players
            for track_id, player in player_dict.items():
                if player.get("has_ball", False):
                    player_color = (255,0,0) # if player has ball, blue highlight
                else:
                    player_color = player.get("team_color",(0,0,255)) # other players, team color

                frame = self.draw_ellipse(frame, player['bbox'], player_color)
                frame = self.draw_rectangle(frame, player['bbox'], player_color, track_id)

            # add annotation for referees (yellow)
            referee_color = (0,255,255)
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], referee_color)

            ball_color = (0,255,0)
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], ball_color)
            
            annotated_frames.append(frame)

        return annotated_frames
