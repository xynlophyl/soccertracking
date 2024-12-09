import cv2
import numpy as np
from utils import get_bottom_center_of_bbox

class ViewTransformer():

    def __init__(self, minimap_keypoints: list, conf: float = 0.5) -> None:

        self.minimap_keypoints = minimap_keypoints
        self.conf = conf

    def get_homography(self, source, target):
        """take source (with confidence values) and target vectors, filter by minimum confidence threshold, then calculate homography transformation"""
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # get indexes with confidence greater than threshold (0.5)
        # TODO: if we want to annotate keypoint detections on VOD, use these indexes instead of array indexes
        self.filtered_indexes = source[:,2] >= self.conf

        # filter detection coordinates
        source = source[self.filtered_indexes][:, :2]

        # filter ground truth coordinates
        target = target[self.filtered_indexes]

        # get homography transformation
        m, _ = cv2.findHomography(source, target)

        return m

    def transform_points(self, points: np.ndarray, m: np.ndarray) -> np.ndarray:
        """uses homography to transform points to 2D plane """

        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, m)

        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_all_points(self, frames, tracks, all_keypoints):
        for frame_num, frame in enumerate(frames):

            # get pitch keypoint detections for current frame
            frame_keypoints = all_keypoints[frame_num][0]

            # get transform
            m = self.get_homography(frame_keypoints, self.minimap_keypoints)

            # get player and ball tracks for current frame
            player_tracks = tracks['players'][frame_num]
            ball_tracks = tracks['ball'][frame_num]

            # get player and ball position coordinates using bounding boxes
            player_positions = [p['bbox'] for p in player_tracks.values()]
            ball_positions = [b['bbox'] for b in ball_tracks.values()]
            # print("pre player_positions", player_positions)
            # print("pre ball_positions", ball_positions)

            player_positions = np.array([get_bottom_center_of_bbox(bbox) for bbox in player_positions])
            ball_positions = np.array([get_bottom_center_of_bbox(bbox) for bbox in ball_positions])
            
            # print("post player_positions", player_positions)
            # print("post ball_positions", ball_positions)

            # transform object positions to 2D plane
            transformed_player_positions = self.transform_points(player_positions, m)
            transformed_ball_positions = self.transform_points(ball_positions, m)

            # update tracks
            for idx, p in enumerate(player_tracks):
                tracks['players'][frame_num][p]['xy_2D'] = transformed_player_positions[idx]

            for idx, b in enumerate(ball_tracks):
                tracks['ball'][frame_num][b]['xy_2D'] = transformed_ball_positions[idx]

        return tracks
    