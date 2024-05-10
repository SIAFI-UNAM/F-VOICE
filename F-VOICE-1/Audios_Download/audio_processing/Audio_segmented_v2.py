# Import necessary libraries
from pydub import AudioSegment
import os
import numpy as np
import noisereduce as nr
from tqdm import tqdm

# Function to segment audio
def segment_audio(input_audio_path, output_folder, s=10):
    # Check if s is between 10 and 15
    if s < 10 or s > 15:
        print("Error: Duration msut be between 10 to 15 seconds.")
        return False
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    # Calculate the number of segments
    lon_a = len(audio)
    milliseconds_u = s * 1000
    segments = int(lon_a / milliseconds_u)
    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)
    for i in tqdm(range(int(segments)), desc="Segmenting audio"):
        audio_i = audio[i*milliseconds_u:milliseconds_u*(i+1)]
        if len(input_audio_path.split('\\')) == 2:
            # Save the segmented audios with the first characters of the audio names
            audio_i.export(os.path.join(output_folder, f"{input_audio_path.split('\\')[1].rstrip(".wav")}_segment_{i}.wav"), format="wav")
        else:
            audio_i.export(os.path.join(output_folder, f"{input_audio_path.split('/')[1].rstrip(".wav")}_segment_{i}.wav"), format="wav")
    # Print a success message
    print("Operation completed.")
    return output_folder
