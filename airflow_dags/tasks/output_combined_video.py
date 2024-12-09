import cv2
import os
import numpy as np
import sys

def output_combined_video(clip_video_path, minimap_video_path, output_path, minimap_size=(300, 400)):
    # Open the clip video
    clip_video = cv2.VideoCapture(clip_video_path)
    minimap_video = cv2.VideoCapture(minimap_video_path)

    # Get properties of the clip video
    clip_width = int(clip_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_height = int(clip_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(clip_video.get(cv2.CAP_PROP_FPS))
    total_frames_clip = int(clip_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize minimap video to the specified size
    minimap_width, minimap_height = minimap_size

    # Output video dimensions (same as clip video)
    output_width = clip_width
    output_height = clip_height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Process each frame
    for frame_idx in range(total_frames_clip):
        ret_clip, frame_clip = clip_video.read()
        ret_minimap, frame_minimap = minimap_video.read()

        if not ret_clip:
            break

        # If minimap video ends earlier, use a black frame
        if not ret_minimap:
            frame_minimap = np.zeros((minimap_height, minimap_width, 3), dtype=np.uint8)
        else:
            # Resize minimap frame to the specified size
            frame_minimap = cv2.resize(frame_minimap, (minimap_width, minimap_height))

        # Overlay minimap frame onto clip frame at the specified position
        x_offset, y_offset = (clip_width - minimap_width) // 2, clip_height - minimap_height
        y_end = y_offset + minimap_height
        x_end = x_offset + minimap_width

        # Ensure the overlay doesn't exceed clip video's frame dimensions
        if y_end > frame_clip.shape[0] or x_end > frame_clip.shape[1]:
            raise ValueError("The overlay position or size exceeds the dimensions of the clip video.")

        roi = frame_clip[y_offset:y_end, x_offset:x_end]
        frame_minimap = cv2.addWeighted(frame_minimap, 0.7, roi, 1 - 0.7, 0)
        # Overlay minimap video onto clip video
        frame_clip[y_offset:y_end, x_offset:x_end] = frame_minimap

        # Write the combined frame to the output video
        out.write(frame_clip)

    # Release all resources
    clip_video.release()
    minimap_video.release()
    out.release()

if __name__ == "__main__":
    input_video_path = sys.argv[1]
    filename = sys.argv[2]
    try:
        GCP_PROJECT_PATH = os.getenv("GCP_PROJECT_PATH", "/home/wwkb1233/airflow/dags/soccertracking")
            
        clip_video_path = f"{GCP_PROJECT_PATH}/outputs/output_annotated_{filename}.avi"  # Path to the first video
        minimap_video_path = f"{GCP_PROJECT_PATH}/outputs/output_minimap_{filename}.avi"  # Path to the second video
        output_path = f"{GCP_PROJECT_PATH}/outputs/output_combined_{filename}.avi"  # Path to save the combined video

        # Define the size and position for minimap video
        minimap_size = (400, 300)  # Width x Height of the resized minimap video

        output_combined_video(clip_video_path, minimap_video_path, output_path, minimap_size)
        print("Video processing complete. Saved as:", output_path)
    except Exception as e:
        print(f"Error outputing file: {e}")
