import os
import cv2
import json
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def extract_keyframes(video_path, output_dir, threshold=30, max_keyframes=10):
    """
    Extract keyframes from a video and save them, with a maximum of max_keyframes.

    :param video_path: Path to the video file
    :param output_dir: Directory to save the keyframes
    :param threshold: Frame difference threshold to determine if a frame is a keyframe
    :param max_keyframes: Maximum number of keyframes to extract
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    keyframe_count = 0

    while cap.isOpened() and keyframe_count < max_keyframes:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count == 1:
            # Save the first frame as a keyframe
            keyframe_path = os.path.join(output_dir, f"keyframe_{keyframe_count:04d}.jpg")
            cv2.imwrite(keyframe_path, frame)
            keyframe_count += 1
        elif prev_frame is not None:
            # Calculate the absolute difference between the current frame and the previous frame
            diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            mean_diff = diff.mean()

            if mean_diff > threshold:
                # Save the keyframe
                keyframe_path = os.path.join(output_dir, f"keyframe_{keyframe_count:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                keyframe_count += 1

        prev_frame = frame

    cap.release()


def extract_audio(video_path, output_audio_path):
    """
    Extract audio from a video and save it as a file.

    :param video_path: Path to the video file
    :param output_audio_path: Path to save the audio file
    """
    dir = os.path.dirname(output_audio_path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            print(f"Warning: The video file {video_path} does not contain an audio stream, skipping audio extraction.")
        else:
            video.audio.write_audiofile(output_audio_path)
    except Exception as e:
        print(f"Failed to extract audio from {video_path}. Error: {e}")



def process_videos_based_on_json(json_file, video_directory):
    """
      Process video files based on the video_id from the JSON file, extracting keyframes and audio.

      :param json_file: Path to the JSON file containing video information
      :param video_directory: Path to the directory containing video files
      """
    # Read the JSON file
    with open(json_file, "r", encoding="utf-8") as f:
        video_ids = [json.loads(line).get("video_id") for line in f if "video_id" in line]

    # Process only the videos specified by video_id in the JSON file
    for video_id in tqdm(video_ids, desc="Processing Videos from JSON"):
        video_path = os.path.join(video_directory, f"{video_id}.mp4")   # Assuming the file extension is .mp4
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            continue

        # Create subdirectories to save the results
        keyframe_output_dir = os.path.join(video_directory, f"keyframes/{video_id}")
        audio_output_file = os.path.join(video_directory, f"audio/{video_id}.mp3")

        # Extract keyframes
        extract_keyframes(video_path, keyframe_output_dir)

        # Extract audio
        extract_audio(video_path, audio_output_file)


if __name__ == '__main__':
    # Path to the JSON file
    DATA = 'FakeSV'
    json_file = f"../data/{DATA}/data.json"  # Replace with the path to the JSON file containing video_id
    # Directory containing video files
    video_directory = f"../data/{DATA}/videos/"  # Replace with the path to the directory containing video files

    # Process videos based on the JSON file
    process_videos_based_on_json(json_file, video_directory)
