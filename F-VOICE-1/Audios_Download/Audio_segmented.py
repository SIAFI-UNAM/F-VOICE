# Import necessary libraries
from pydub import AudioSegment
import os
import numpy as np
import noisereduce as nr

# Function to segment audio
def segment_audii(input_audio_path, output_folder, s=10):
    # Check if s is between 10 and 15
    if s < 10 or s > 15:
        print("Error: La duración de los segmentos debe estar entre 10 y 15 segundos.")
        return False
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    # Convert to numpy array
    audio_np = np.array(audio.get_array_of_samples())
    # Get the frame rate (sample rate) of the audio
    sr = audio.frame_rate
    # Reduce noise
    reduced_noise = nr.reduce_noise(audio_np, sr)
    # Convert back to AudioSegment
    audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # Calculate the number of segments
    lon_a = len(audio)
    milliseconds_u = s * 1000
    segments = lon_a / milliseconds_u
    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)
    for i in range(int(segments)):
        audio_i = audio[i*milliseconds_u:milliseconds_u*(i+1)]
        # Save the segmented audios with the first characters of the audio names
        audio_i.export(os.path.join(output_folder, f"{input_audio_path.split('/')[-1][:5]}_segment_{i}.wav"), format="wav")
    # Print a success message
    print("La operación fue realizada con éxito.")
    return output_folder
# Function to display user options
