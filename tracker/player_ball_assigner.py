import sys
sys.path.append('../')

from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():

    def __init__(self):
        self.max_player_ball_distance = 70 # max pixels away from ball a player can be to get assigned  

    def _assign_ball_to_player(self, players: list, ball: list):

        # use center of ball
        ball_position = get_center_of_bbox(ball)

        # assign player with ball possession (if exists)
        min_distance = float('inf')
        assigned_player = -1
        for player_id, player in players.items():
            # get player position as bbox
            x1, y1, x2, y2 = player['bbox']
            
            # calculate distance betwee  
            distance_left = measure_distance((x1, y2), ball_position)
            distance_right = measure_distance((x2, y2), ball_position)
            distance = min(distance_right, distance_left)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player

    def assign_ball_to_player(self, tracks):

        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self._assign_ball_to_player(player_track, ball_bbox)

            # add has_ball property to tracking data if assignment is found
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True

        return tracks
