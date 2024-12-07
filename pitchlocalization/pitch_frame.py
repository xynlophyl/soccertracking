import copy
import cv2
# from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import numpy as np

class PitchFrame():

    def __init__(self):

        # TODO: fill with desired pitch parameters (e.g., pitch_color, line_color)
            # ONLY CHANGE FIGSIZE AFTER TRANSFORMATION TO IMAGE ARRAY, as changing during pitch.draw() will affect scaling

        self.pitch_frame = self._get_pitch()

    # def _get_pitch(self):

    #     """
    #     returns np representation of mplsoccer pitch plot
    #     """

    #     pitch = Pitch(
    #         pitch_type='custom',
    #         pitch_length=105, pitch_width=68,
    #         pitch_color='#aabb97',
    #         line_color='#c7d5cc',
    #     )

    #     fig, ax = pitch.draw()

    #     # update plot
    #     fig.canvas.draw()

    #     # save plot to numpy array using rgba buffer
    #     pitch_frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    #     width, height = fig.canvas.get_width_height()
    #     # pitch_frame = pitch_frame.reshape((height * 2, width * 2) + (4,))[:,:,:3] # reshape to rgb, remove alpha
    #     pitch_frame = pitch_frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3] # reshape to rgb, remove alpha

    #     plt.close(fig)

    #     return np.ascontiguousarray(pitch_frame, dtype=np.uint8)

    def _get_pitch(self):

        """
        returns np representation of mplsoccer pitch plot
        """

        pitch_frame = cv2.imread('./assets/mplpitch.png')

        return pitch_frame

    def get_pitch_frame(self):

        pitch_frame = copy.deepcopy(self.pitch_frame)

        return pitch_frame

    def draw_circle(self, frame, xy, color):

        # draw circle to represent player
        frame = cv2.circle(frame, xy, radius=5, color=color, thickness=-1)

        return frame

    def add_circle_text(self, frame, xy, text, color):

        """
        add text inside circle annotation
        """

        # Get the size of the text so we can center it inside the circle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, thickness = 2)

        # calculate center position of circle
        text_x = xy[0] - text_height // 4
        text_y = xy[1] - text_height // 4

        # put text inside the circle
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = color, # TODO: make color always visible depending on team assignment (if team color is black, then text won't show up)
            thickness = 2
        )

        return frame

    def add_circle_highlight(self, frame, xy):

        """
        highlight circle with center xy
        """

        # TODO

        return frame


    def draw_annotations(self, frames, tracks, use_jersey_numbers = False):

        """
        create 2D pitch visualization for each frame in original VOD
        TODO: not sure if VOD frames are necessary
        """

        all_pitch_frames = []

        from tqdm.auto import tqdm

        for frame_num, frame in tqdm(enumerate(frames)):

            pitch_frame = self.get_pitch_frame()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw players on pitch
            for track_id, player in player_dict.items():
                player_color = player.get("team_color", (0,0,255))
                if use_jersey_numbers:
                    track_id = player.get("jn", "x")
                x, y = player['xy_2D']
                xy = (int(x), int(y)) # TODO: scale coordinates to fit frame dimensions (from plot coordinates)
                pitch_frame = self.draw_circle(pitch_frame, xy, player_color)
                pitch_frame = self.add_circle_text(pitch_frame, xy, str(track_id), player_color) # TODO: if player identification works out, use that instead of track_id
                pitch_frame = self.add_circle_highlight(pitch_frame, xy) # TODO: add highlight around player if in possession of ball (maybe? if it doesn't clutter visualization too much)

            ball_color = (0,0,0) # TODO: change ball color or design?
            for track_id, ball in ball_dict.items():
                xy = ball['xy_2D']
                xy = (int(x*640/100), int(y*480/100)) # TODO: scale coordinates to fit frame dimensions (from plot coordinates)
                pitch_frame = self.draw_circle(pitch_frame, xy, ball_color)

            all_pitch_frames.append(pitch_frame)

        return all_pitch_frames
