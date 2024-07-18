import json
import os
import subprocess

json_path = 'data/TAPOS/tapos_annotation.json'
video_folder = 'data/TAPOS/rawvideo'
output_folder = 'data/TAPOS/trim_video'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(json_path, 'r') as f:
    annotations = json.load(f)


for video_id, segments in annotations.items():
    for segment_id, segment_info in segments.items():
        start_offset = float(segment_id.split('_')[1] + '.' + segment_id.split('_')[2])
        end_offset = float(segment_id.split('_')[3] + '.' + segment_id.split('_')[4])
        base_time = segment_info['shot_timestamps'][0]
        print(start_offset,end_offset,segment_id)
        assert 1==2

        start_time = base_time + start_offset
        end_time = base_time + end_offset

        output_filename = f"{video_id}_{segment_id}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        input_path = os.path.join(video_folder, f"{video_id}.mp4")

        if not os.path.exists(input_path):
            continue

        command = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            output_path
        ]

        try:

            subprocess.run(command, check=True)
            print(f"trim video: {video_id} to {output_filename}successfully")
        except subprocess.CalledProcessError as e:
            print(f"trim video: {video_id} error: {e}")