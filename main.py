# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install git+https://github.com/openai/whisper.git
!pip install torch

import whisper
import torch
import os


# Check if a CUDA-enabled GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model and move it to the GPU if available
model = whisper.load_model("large", device=device)

# Specify the path to the audio file on Google Drive
audio_file = "/content/drive/MyDrive/awsbc9_source/1_2_nerwork_101.wav"

# Set the input language to Turkish
input_language = "tr"  # Turkish language code

# Transcribe the entire audio file with fp16 enabled and specified language
result = model.transcribe(audio_file, fp16=False, language=input_language)

# Helper function to convert seconds to SRT timestamp format
def format_timestamp(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Create the SRT file content
srt_content = []
for i, segment in enumerate(result["segments"]):
    start_time = format_timestamp(segment["start"])
    end_time = format_timestamp(segment["end"])
    text = segment["text"].strip()
    srt_content.append(f"{i + 1}")
    srt_content.append(f"{start_time} --> {end_time}")
    srt_content.append(text)
    srt_content.append("")

# Write the SRT file to Google Drive
output_srt_file = "/content/drive/MyDrive/awsbc9_source/1_2_nerwork_101.srt"
with open(output_srt_file, "w") as f:
    f.write("\n".join(srt_content))

print(f"Subtitle file saved to {output_srt_file}")

