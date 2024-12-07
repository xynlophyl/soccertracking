# soccertracking

An implementation of soccer match game state reconstruction, inspired by the [SoccerNet GSR Challenge](https://www.soccer-net.org/tasks/game-state-reconstruction).

This project utilizes a multimodal architecture to track key objects (players, referees, ball, pitch keypoints) and perform various annotations and visualizations. Functionality comprises of object detection (YOLOv5-detect), tracking (roboflow supervision), team assignment (k-means clustering) jersey number detection (MMOCR), pitch keypoint detection (YOLOv5-pose; TVCalib) and pitch localization (homography transformations).

## References
- https://arxiv.org/abs/2404.11335


TODO
    - object detection: tracking, color/team assignments, ball position interpolation
    - pitch localization: camera movement estimators, perspective transformations, position smoothing
    - visualizations: raw annotations, 2D minimap view, player heatmaps, speed estimators 