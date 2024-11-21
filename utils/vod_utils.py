import cv2
import json
import os
from tqdm import tqdm
import uuid

def read_video(video_path: str):

    """
    load video as invididual frames
    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def save_video(frames: list, outpath: str):

    """
    stitch frames into video output
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        outpath,
        fourcc,
        24,
        (frames[0].shape[1], frames[0].shape[0])
    )

    for frame in frames:
        out.write(frame)
    
    out.release()

def generate_uuid() -> str:

    return uuid.uuid4().hex

def create_manifest(metadata: dict, outpath: str) -> None:
    
    """
    create manifest file to store chunk metadata for future reconstruction
    """

    with open(outpath, 'w') as f:
        json.dump(
            {'chunks': metadata}, 
            f, 
            indent = 4
        )

def split_vod(input_path: str, outpath: str, duration: int = 30) -> None:
    
    """
    splits match vod into smaller video chunks (default 30s)
    """

    # check if outpath exists
    if not os.path.exists(outpath):
        raise Exception(f"Path Error: output file path {outpath} does not exist")

    # capture video file into individual frames
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Error: Video could not be opened")
    
    # get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # get total frames in one chunk
    frames_per_chunk = int(fps*duration)
    
    chunks = [] # metadata

    # get total frames in video
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # init progress bar
    bar = tqdm(total = nframes)
    
    chunk_num = 1 # current chunk iteration
    frame_num = 0 # current frame iteration

    while True:

        # get current video frame
        ret, frame = cap.read() 
        if not ret: 
            break   
            
        if frame_num == 0:

            # init chunk and metadata
            chunk_uuid = str(uuid.uuid4()) #  chunk uuid
            chunk_file  = os.path.join(outpath, f'{chunk_uuid}.mp4') # output file
            out = cv2.VideoWriter(
                chunk_file,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (int(cap.get(3)), int(cap.get(4)))
            )

            # create chunk metadata
            metadata = {
                'uuid': chunk_uuid,
                'filename': chunk_file,
                'order': chunk_num,
            }

            chunks.append(metadata)

        out.write(frame)
        frame_num += 1

        if frame_num >= frames_per_chunk:

            # release chunk output
            out.release()
            
            # reset chunk iterations
            chunk_num += 1
            frame = 0 

        # update progress
        bar.update(1)

    # release last video chunk
    if out is not None:
        out.release()
    cap.release()

    # create manifest file
    input_file = os.path.split(input_path)[-1]
    input_name, _ = os.path.splitext(input_file)
    manifest_path = os.path.join(outpath, f'{input_name}_manifest.json')
    create_manifest(metadata, manifest_path)

# TODO: implement vod construction with chunks and metadata 
def reconstruct_vod(manifest_file: str):

    """
    reconstruct full vod using shards and metadata
    TODO: want to be able to reconstruct shards with model tracking labels
        this means that the inpt videos will be different than the file paths stored in metadata
    """

    pass