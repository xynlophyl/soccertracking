from ultralytics import YOLO
import supervision as sv
import os
import pandas as pd
import pickle

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
        
        return detections

    def get_object_tracks(
            self, 
            frames: list, 
            remap_gk: bool = True,
            read_from_stub: bool = False,
            stubpath: str = None
        ) -> list:

        """
        assigns tracking for each detected object in video across across each frame, with conditional remapping of goalkeepers to general player class
        """

        # if stub already exists, return
        if read_from_stub and stubpath is not None and os.path.exists(stubpath):
            with open(stubpath, 'rb') as f:
                tracks = pickle.load(f)

            return tracks

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        detections = self.detect_frames(frames)

        for frame_id, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # convert to supervision detction format
            detections_sv = sv.Detections.from_ultralytics(detection)

            if remap_gk:
                # convert goalkeeper to player object
                for obj_idx, class_id in enumerate(detections_sv.class_id):
                    
                    if cls_names[class_id] == "goalkeeper":
                        detections_sv.class_id[obj_idx] = cls_names_inv["player"]
            
                # update detection mappings
                detection_with_tracks = self.tracker.update_with_detections(detections_sv)

            # add current bbox detections of each object according to their track_id
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # tracking players and referees in each frame
            for frame_detection in detection_with_tracks:

                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_id][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_id][track_id] = {"bbox": bbox}

            # tracking ball
            for frame_detection in detections_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_id][0] = {"bbox": bbox}

        if stubpath is not None:
            with open(stubpath, 'wb') as f:
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

    def draw_annotations(self, frames: list, tracks: list):

        """
        draw custom annotations using bbox detections
        """

        output_frames = []

        for frame_id, frame in enumerate(frames):

            frame = frame.copy()

            player_dict = tracks["players"][frame_id]
            referee_dict = tracks["referees"][frame_id]
            ball_dict = tracks["ball"][frame_id]
            
            for track_id, player in player_dict.items():
                x1, y1, x2, y2 = player['bbox']

        # TODO: finish implementing annotations for visualizing detection boxes
