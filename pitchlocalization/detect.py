import cv2
import os
import pickle
from ultralytics import YOLO

class KeypointDetector():

    def __init__(self, model_path: str):

        self.model = YOLO(model_path)

    def detect_pose(
            self,
            frames: list,
            batch_size: int = 16
    ):
        
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            detections = self.model.predict(batch, conf = 0.3)
            all_detections.extend(detections)
        
        return all_detections
    
    def get_keypoints(
            self,
            frames: list,
            read_from_stub: bool | None = None,
            stub_path: bool | None = None
    ):
        
        # check for existing keypoint stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                all_pitch_keypoints = pickle.load(f)
                return all_pitch_keypoints
            
        # get keypoint detections, with confidence values in batches
        detections = self.detect_pose(frames)
        all_pitch_keypoints = []
        for detection in detections:
            all_pitch_keypoints.append(detection.keypoints[0].data.cpu().numpy())
        
        # create stub file
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(all_pitch_keypoints, f)

        return all_pitch_keypoints
    
    def draw_video_annotations(self, frames: list, all_pitch_keypoint_detections: list):

        """
        draw annotations onto vod frames to visualize pitch keypoints
        """
        assert len(frames) == len(all_pitch_keypoint_detections)

        for frame_num, frame in enumerate(frames):

            pitch_keypoints = all_pitch_keypoint_detections[frame_num]

            for i, (x,y, _) in enumerate(pitch_keypoints):
                x, y = int(x), int(y)
                cv2.circle(frame, (x,y), radius=5, color=(255, 255, 255), thickness=-1)
                cv2.putText(
                    frame,
                    f"{i}",
                    (x,y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2
                )

        return frames
    
