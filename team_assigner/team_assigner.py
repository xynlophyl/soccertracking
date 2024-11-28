from sklearn.cluster import KMeans
import random
from utils import calculate_centroid, measure_distance, get_center_of_bbox

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            if "cls_name" in player_detection and player_detection["cls_name"] == "player":
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame,bbox)
                player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # self.team_colors[1] = kmeans.cluster_centers_[0]
        # self.team_colors[2] = kmeans.cluster_centers_[1]
        self.team_colors[1] = [189, 0, 255] # hard coded team 1 color, in BGR
        self.team_colors[2] = [240, 255, 160] # hard coded team 2 color, in BGR


    def get_player_team(self,frame, player_track, player_id):
        player_bbox = player_track['bbox']
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1 # team 1 or 2

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_goalkeeper_team(self, frames, tracks):
        """TODO goalkeeper assignment isn't always correct
                consider using left/right heuristic for team assignment instead (goalkeeper assignment can be easy), and see how to map 0/1 values to l/r for players """

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                if track["cls_name"] == "player":
                    team = self.get_player_team(frames[frame_num],   
                                                        track,
                                                        player_id)
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team]

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
                        tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[1]
                    else:
                        tracks['players'][frame_num][player_id]['team'] = 2
                        tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[2]

        return tracks

    def assign_team_by_sampling(self, frames, tracks):
        for _ in range(100):
            random_frame_index = random.randrange(len(frames))
            team_color_counter = [0, 0]
            self.player_team_dict = {}
            self.assign_team_color(frames[random_frame_index], tracks['players'][random_frame_index])
            for player_id, track in  tracks['players'][random_frame_index].items():
                if track["cls_name"] == "player":
                    team = self.get_player_team(
                        frames[random_frame_index],   
                        track,
                        player_id
                    )
                    team_color_counter[team-1] += 1

            total_players = sum(team_color_counter)
            smaller_team = min(team_color_counter)
            if smaller_team > total_players * 0.4: # if the smaller team size is at least 40% of the total, the proportions should make sense
                break
            else:
                print("k-mean too biased. re-running")

        return tracks
