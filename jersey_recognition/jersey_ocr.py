import pandas as pd
import pickle
import numpy as np
from tqdm.auto import tqdm
import torch
import os

from sn_gamestate.jersey.mmocr_api import MMOCR
from sn_gamestate.jersey.voting_tracklet_jn_api import VotingTrackletJerseyNumber
from tracklab.utils.attribute_voting import select_highest_voted_att

class JerseyOCR():

    def __init__(self, batch_size: int, device: str):

        self.batch_size = batch_size

        self.mmocr = MMOCR(
            batch_size = batch_size,
            device = device
        )

        self.vtjn = VotingTrackletJerseyNumber(
            cfg = None,
            device = device
        )

    @torch.no_grad()
    def load_tracks_YOLO(self, frames, tracks):
        """
        returns tracking detections with video frame metadata as Pandas dataframe
        """
        detections = []

        for frame_num, frame in enumerate(frames):

            for player_id, player_track in tracks['players'][frame_num].items():
                # get detected player's bbox
                x1, y1, x2, y2 = player_track['bbox']
                detected_bbox = frame[int(y1):int(y2),int(x1):int(x2)]


                track = {
                    'player_id': player_id,
                    'frame_id': frame_num,
                    'detected_bbox': detected_bbox
                }

                detections.append(track)

        detections_df = pd.DataFrame(detections)

        return detections_df

    # @torch.no_grad()
    # def get_jersey_numbers(
    #         self,
    #         frames: list,
    #         tracks: list,
    #         read_from_stub: bool,
    #         stub_path: str | None = None
    #     ):

    #     if read_from_stub and stub_path is not None and os.path.exists(stub_path):
    #         tracks = pd.read_pickle(stub_path)

    #         return tracks

    #     detections_df = self.load_tracks_YOLO(frames, tracks)

    #     jersey_number_detection = []
    #     jersey_number_confidence = []

    #     for b in tqdm(range(0, len(detections_df), self.batch_size)):

    #         batch = [detection for detection in detections_df.loc[b:b+self.batch_size-1, 'detected_bbox']]

    #         predictions = self.mmocr.run_mmocr_inference(batch)

    #         for pred in predictions:
    #             jn, conf = self.mmocr.extract_jersey_numbers_from_ocr(pred)

    #             jersey_number_detection.append(jn)
    #             jersey_number_confidence.append(conf)

    #     detections_df.loc[:, 'jersey_number_detection'] = jersey_number_detection
    #     detections_df.loc[:, 'jersey_number_confidence'] = jersey_number_confidence

    #     return detections_df

    @torch.no_grad()
    def process(
            self,
            detections
        ):

        jersey_number_detection = []
        jersey_number_confidence = []

        for b in tqdm(range(0, len(detections), self.batch_size)):

            batch = [detection for detection in detections.loc[b:b+self.batch_size-1, 'detected_bbox']]

            predictions = self.mmocr.run_mmocr_inference(batch)

            for pred in predictions:
                jn, conf = self.mmocr.extract_jersey_numbers_from_ocr(pred)

                jersey_number_detection.append(jn)
                jersey_number_confidence.append(conf)

        detections.loc[:, 'jersey_number_detection'] = jersey_number_detection
        detections.loc[:, 'jersey_number_confidence'] = jersey_number_confidence

        return detections
    
    @torch.no_grad()
    def get_jersey_tracks(self, frames, tracks, read_from_stub, stub_path): 
        """
        TODO: might need to refactor to make the workflow cleaner 
            i.e. the tracks from stub_path has a different format from the format of tracks given as input, so it might be confusing for which format tracks is currently using
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                
            return tracks
        
        # convert tracks to soccernet format
        detections_df = self.load_tracks_YOLO(frames, tracks)

        # get jersey detections
        detections_df = self.process(detections_df)

        # assigning constant jersey number using majority voting
        detections_df = self.vote_jersey_number(detections_df)

        # convert back to original tracks format
        tracks = self.add_jersey_numbers_to_tracks(detections_df, tracks)

        # save jersey
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def vote_jersey_number(self, detections):
        detections["jn_tracklet"] = [np.nan] * len(detections)
        if "player_id" not in detections.columns:
            return detections
        for player_id in detections.player_id.unique():

            tracklet = detections[detections.player_id == player_id]
            jersey_numbers = tracklet.jersey_number_detection
            jn_confidences = tracklet.jersey_number_confidence

            tracklet_jn = [select_highest_voted_att(jersey_numbers,
                                                    jn_confidences)] * len(tracklet)
            detections.loc[tracklet.index, "jn_tracklet"] = tracklet_jn

        return detections

    def add_jersey_numbers_to_tracks(self, detections, tracks):
        detected_jerseys = detections[detections.jn_tracklet.notna()]

        for idx in detected_jerseys.index:

            row = detections.loc[idx]
            frame_id= row['frame_id']
            player_id = row['player_id']

            tracks['players'][frame_id][player_id]['jn'] = row['jn_tracklet']
            tracks['players'][frame_id][player_id]['jn_detection'] = row['jersey_number_detection']
            tracks['players'][frame_id][player_id]['jn_confidence'] = row['jersey_number_confidence']

        return tracks

    def save_tracks(self, tracks, stub_path):
        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
