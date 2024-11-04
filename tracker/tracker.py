from ultralytics import YOLO
import supervision as sv
import os
import pickle

class Tracker():

    """
    object detection model with tracking
    """

    def __init__(self, model_path):
        
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(
        self,
        frames: list,
        batch_size: int = 32,
    ):
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
    ):

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

        for f, detection in enumerate(detections):

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
                    tracks["players"][f][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][f][track_id] = {"bbox": bbox}

            # tracking ball
            for frame_detection in detections_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][f][0] = {"bbox": bbox}

        if stubpath is not None:
            with open(stubpath, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks