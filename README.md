# soccertracking

An implementation of soccer match game state reconstruction, inspired by the [SoccerNet GSR Challenge](https://www.soccer-net.org/tasks/game-state-reconstruction).

<br>
<p align="center">
    <img src="https://github.com/user-attachments/assets/6f7d0af1-dcb1-4380-afc6-599b72c15a58" alt="tracking-annotations" height="270" width="480">
    <img src="https://github.com/user-attachments/assets/33f7a6aa-7e2b-4005-8b65-bfe1b0c578f2" alt="2D minimap" height="270" width="480">
</p>
<br>

This project uses a multimodal architecture to track key objects (players, referees, ball, pitch keypoints) and perform various annotations and visualizations. Functionality  comprises of:
- object detection: YOLOv5-detect
- tracking: roboflow supervision
- team assignment: k-means clustering
- jersey number detection: MMOCR
- pitch keypoint detection: YOLOv5-pose; TVCalib
- pitch localization: homography transformations

## References
- https://arxiv.org/abs/2404.11335

<br>
TODO
- object detection: tracking, color/team assignments, ball position interpolation
- pitch localization: camera movement estimators, perspective transformations, position smoothing
- visualizations: raw annotations, 2D minimap view, player heatmaps, speed estimators 
