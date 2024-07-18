import os
import subprocess

def extract_frames(video_path, frame_output_dir):
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-f', 'image2',
        '-qscale:v', '2',
        '-loglevel', 'quiet',
        os.path.join(frame_output_dir, 'frame%d.jpg')
    ]
    subprocess.run(command, check=True)

def process_videos(video_root, frame_root):
    for video_file in os.listdir(video_root):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_root, video_file)
            video_path = os.path.join(video_root, video_file)
            video_name = os.path.splitext(video_file)[0]
            frame_output_dir = os.path.join(frame_root, video_name)
            extract_frames(video_path, frame_output_dir)
            print(f"Extracted frames from {video_path} to {frame_output_dir}")

if __name__ == "__main__":
    video_root = 'data/TAPOS/trim_video' 
    frame_root = 'data/TAPOS/images/train'
    process_videos(video_root, frame_root)