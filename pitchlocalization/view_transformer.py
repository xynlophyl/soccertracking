import cv2
import numpy as np

class ViewTransformer():

    def __init__(self, source: np.ndarray, target: np.ndarray, conf: float = 0.5) -> None:

        """
        take source (with confidence values) and target vectors, filter by minimum confidence threshold, then calculate homography transformation 
        """

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # get indexes with confidence greater than threshold (0.5)
        # TODO: if we want to annotate keypoint detections on VOD, use these indexes instead of array indexes
        self.filtered_indexes = source[:,2] >= conf

        # filter detection coordinates
        source = source[self.filtered_indexes][:, :2]

        # filter ground truth coordinates
        target = target[self.filtered_indexes]

        # get homography transformation
        self.m, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:

        """
        uses homography to transform points to 2D plane 
        """

        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, self.m)

        return transformed_points.reshape(-1, 2).astype(np.float32)
